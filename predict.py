import cv2

import click 
import torch as th 

from loguru import logger 
from libraries.strategies import * 
from torchvision import transforms as T 

@click.command()
@click.option('--source', help='image source data', type=click.Path(True))
@click.option('--generator', help='path to generator', type=click.Path(True))
def upscale(source, generator):
	I = read_image(source, by='th').float()
	I = I / th.max(I) 
	I = T.Normalize(mean=[0.5]*3, std=[0.5]*3)(I)
	G = th.load(generator, map_location=th.device('cpu')).eval()
	J = G(I[None, ...])
	R = to_grid(J, nb_rows=1)
	O = th2cv(R) 
	cv2.imshow('000', O)
	cv2.waitKey(0)

if __name__ == '__main__':
	logger.debug('...[infenrec] ...')
	upscale()