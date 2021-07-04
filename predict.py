import cv2

import click 
import torch as th 
import torch.nn as nn 

from loguru import logger 
from libraries.strategies import * 
from torchvision import transforms as T 

@click.command()
@click.option('--source', help='image source data', type=click.Path(True))
@click.option('--generator', help='path to generator', type=click.Path(True))
def upscale(source, generator):
	X = T.Resize((64, 64))(read_image(source, by='th').float())
	I = X / th.max(X)
	G = th.load(generator, map_location=th.device('cpu')).eval()
	J = G(I[None, ...])
	R = to_grid(J, nb_rows=1)
	O = th2cv(R) 
	P = th2cv(nn.functional.interpolate(I[None, ...], scale_factor=4)[0] )

	H, W = O.shape[:2]
	cv2.imshow('000', O)
	cv2.imshow('001', np.hstack([O, P]) )
	cv2.waitKey(0)

if __name__ == '__main__':
	logger.debug('...[infenrec] ...')
	upscale()