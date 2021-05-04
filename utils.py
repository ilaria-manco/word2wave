class MinMaxScaler():
    """
    Transforms each channel to the range [0, 1].
    """    
    def __call__(self, tensor):
        for ch in tensor:
            scale = 1.0 / (ch.max(dim=0)[0] - ch.min(dim=0)[0])        
            ch.mul_(scale).sub_(ch.min(dim=0)[0])        
        return tensor

def sample_noise(size, latent_dim=100):
    noise = torch.FloatTensor(size, latent_dim)
    noise.data.normal_()
    return noise

def latent_space_interpolation(generator, n_samples=10, source=None, target=None):
  if source is None and target is None:
    random_samples = sample_noise(2, 100)
    source = random_samples[0]
    target = random_samples[1]
  with torch.no_grad():
    interpolated_z = []
    for alpha in np.linspace(0, 1, n_samples):
      interpolation = alpha * source + ((1 - alpha) * target)
      interpolated_z.append(interpolation)

    interpolated_z = torch.stack(interpolated_z)
    generated_audio = generator(interpolated_z)
  return generated_audio
