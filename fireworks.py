import math
import random
import cv2
import numpy as np
import scipy.ndimage

class Matter:
	def __init__(self, pos=None, vel=None):
		#if pos is None:
		self.pos = np.zeros((3,)) #np.random.random((3,))-0.5
		self.vel = (np.random.random((3,))-0.5)*5+(-0.5,0,0)
		self.drift = 0.125
	def update_velocity(self, force, time_delta):
		self.vel += np.dot(time_delta, force)
	def update_position(self, time_delta):
		self.pos += np.dot(time_delta, self.vel+np.random.random(self.vel.shape)*self.drift)

class Universe:
	def __init__(self, N=50, time_delta=0.01, gravity=(2.0,0,0)):
		self.particles = [Matter() for p in range(N)]
		self.time_delta = time_delta
		self.gravity = gravity
		self.pointimg = 1.5*cv2.imread('Point.png', 0).astype('float32')/255.0
		self.img = np.zeros((320, 320, 3))
		self.imgs = []
		self.scale = float(self.img.shape[0] / 2)
		self.t = 0
	
	def reset(self):
		self.particles = [Matter() for p in range(len(self.particles))]
		self.t = 0
	
	def clock(self):
		for p in self.particles:
			p.update_velocity(self.gravity, self.time_delta)
			p.update_position(self.time_delta)
		self.t += self.time_delta
			
	def display(self, camerapos=(0,0,2000)):
		#self.img = np.zeros((256, 256, 3))
		self.img *= 0.85
		for p in self.particles:
			# Render audio based on position and 
			
			# Render white noise surrounding each particle, including a pressure wave exerted by each particle on the observer.
			
			# Now render each particle.
			if p.pos[2] - 1 >= camerapos[2]:
				continue
			dz = (camerapos[2] - p.pos[2])
			scale = (1+0.05*np.random.random())*400.0/dz
			#scale = 500.0/dz
			renderimg = scipy.ndimage.zoom(self.pointimg, scale*(1.0+np.random.random()))
			h,w = self.img.shape[0:2]
			rh,rw = renderimg.shape[0:2]
			px,py = int(w*(scale*p.pos[1]+0.5)),int(h*(scale*p.pos[0]+0.5))
			l,r,u,b = int(px-rw/2),int(px+rw/2),int(py-rh/2),int(py+rh/2)
			if l < 0:
				renderimg = renderimg[:,-l:]
				l = max(0, l)
			if r >= self.img.shape[1]:
				renderimg = renderimg[:,:-(r-self.img.shape[1])]
				r = min(self.img.shape[1]-1, r)
			if u < 0:
				renderimg = renderimg[-u:,:]
				u = max(0, u)
			if b >= self.img.shape[0]:
				renderimg = renderimg[:-(b-self.img.shape[0]),:]
				b = min(self.img.shape[0]-1, b)
			#print(renderimg.shape, px,py, l,r,u,b)
			if l < r and u < b and np.product(renderimg[:b-u,:r-l].shape) > 0:
				#print(l,r,u,b)
				# TODO Still an indexing bug, off-by-one:
				#for c in range(3):
				c = np.random.randint(0,3)
				self.img[u:b,l:r,c] += 0.5*math.exp(-self.t)*np.linalg.norm(p.vel, ord=2)*(0.5+2*np.random.random())*renderimg[:b-u,:r-l]
		#self.imgs.append(self.img)
		#if len(self.imgs) > 10:
		#	del self.imgs[0]
		#img = np.mean(self.imgs, axis=0)
		img = self.img
		cv2.imshow('Fireworks', img)
		#cv2.waitKey(1)

u = Universe()
while True:
	u.clock()
	u.display()
	k = cv2.waitKey(1)
	if k == 27:
		break
	elif k == 32:
		u.reset()
	