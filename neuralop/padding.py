from torch.nn import functional as F
from torch import nn

class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float
        typically, between zero and one, percentage of padding to use
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """
    def __init__(self, domain_padding, padding_mode='one-sided', output_scaling_factor=None):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        self.output_scaling_factor = output_scaling_factor

        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()
        
        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)
    
    def pad(self, x):
        """Take an input and pad it by the desired fraction
        
        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)]*len(resolution)
        if self.output_scaling_factor is None:
            self.output_scaling_factor = [1]*len(resolution)
        elif isinstance(self.output_scaling_factor, (float, int)):
            self.output_scaling_factor = [float(self.output_scaling_factor)]*len(resolution)

        try:
            padding = self._padding[f'{resolution}']
            return F.pad(x, padding, mode='constant')

        except KeyError:
            padding = [int(round(p*r)) for (p, r) in zip(self.domain_padding, resolution)]
            
            print(f'Padding inputs of {resolution=} with {padding=}, {self.padding_mode}')
            
            
            output_pad = padding

            output_pad = [int(round(i*j)) for (i,j) in zip(self.output_scaling_factor,output_pad)]
            
            
            # the F.pad(x, padding) funtion pads the tensor 'x' in reverse order of the "padding" list i.e. the last axis of tensor 'x' will be
            # padded by the amount mention at the first position of the 'padding' vector.
            # The details about F.pad can be found here : https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

            if self.padding_mode == 'symmetric':
                # Pad both sides
                unpad_indices = (Ellipsis, ) + tuple([slice(p, -p, None) for p in output_pad[::-1] ])
                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == 'one-sided':
                # One-side padding
                unpad_indices = (Ellipsis, ) + tuple([slice(None, -p, None) for p in output_pad[::-1]])
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f'Got {self.padding_mode=}')
            
            self._padding[f'{resolution}'] = padding
            

            padded = F.pad(x, padding, mode='constant')

            out_put_shape = padded.shape[2:]

            out_put_shape = [int(round(i*j)) for (i,j) in zip(self.output_scaling_factor,out_put_shape)]

            self._unpad_indices[f'{[i for i in out_put_shape]}'] = unpad_indices

            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs
        """
        unpad_indices = self._unpad_indices[f'{list(x.shape[2:])}']
        return x[unpad_indices]
    
