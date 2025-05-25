import torch

from utils.utils_lgv.subspaces import Subspace
from utils.utils_lgv.subspace_inference_utils import flatten, set_weights


class SWAG(torch.nn.Module):

    def __init__(self, base, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()

        self.base_model = base
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'
        
    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.model_device = device.type
        self.subspace.to(device=torch.device('cpu'), dtype=dtype, non_blocking=non_blocking)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([param.detach().cpu() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        set_weights(self.base_model, self.mean, self.model_device)

    def sample(self, scale=0.5, diag_noise=True, cov_factor=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        z = torch.zeros_like(variance)
        if cov_factor:
            eps_low_rank = torch.randn(self.cov_factor.size()[0])
            z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance.sqrt() * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()
