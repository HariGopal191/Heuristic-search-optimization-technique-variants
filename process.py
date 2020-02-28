# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:47:58 2020

@author: HariGopal V
"""

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import cv2
import numpy as np
from skimage.morphology import watershed
from scipy import ndimage
from skimage.feature import peak_local_max
import argparse
import random
from PSO import PSO

#--- IMAGE CLASS --------------------------------------------------------------+

class Image:

    def __init__(self, path, output):
        self.path = path
        self.output = output
        self.image = None
        self.labels = None
        self.X = []
        self.Y = []
        self.population = []
        self.chromosomes = []
        self.fitnesses = []
        self.max_iter = 100
        self.iter = 0
        self.fixed = random.randint(0, 9)
        self.threshold = 0
        self.contours = None
        self.reg = 0

    #   reading/aquiring the scan image
    def acquisition(self):
        self.image = cv2.imread(self.path)
        self.Pre_Processing()

    #   an empty function for the requirement of thresholding
    def nothing(self, x):
        pass

    #   set a manual threshold value
    def thresholding(self):
        print('Press Space key once caliberated!')
        name = 'Applying Threshold'
        cv2.namedWindow(name)
        cv2.createTrackbar('Threshold', name, 0, 255, self.nothing)
        while(1):

            thresh = cv2.getTrackbarPos('Threshold', name)

            th3, image = cv2.threshold(self.image,thresh,255,cv2.THRESH_TOZERO)

            cv2.imshow(name, image)

            k = cv2.waitKey(5) & 0xFF
            if k == ord(' '):
                cv2.destroyWindow(name)
                return thresh

    #   Pre_process the image
    def Pre_Processing(self):
        # convert the image to gray if coloured and resize the image to (224, 224) pixels
        self.image = cv2.resize(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (224, 224), interpolation = cv2.INTER_NEAREST)
        thresh = self.thresholding()
        th3, self.image = cv2.threshold(self.image,thresh,255,cv2.THRESH_TOZERO)    # applying threshold with a manual threshold value
        cv2.imshow("thresh", self.image)    # show the output image
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.De_Noising()

    #   MedianBlur denoising for removing of salt-pepper noise
    def De_Noising(self):
        self.image = cv2.medianBlur(np.float32(self.image), 5)
        # show the output image
        cv2.imshow("Blur", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.segment()

    #   Watershed segmentation
    def segment(self):
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(self.image)
        localMax = peak_local_max(D, indices=False, min_distance=75, labels=self.image)
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        self.labels = watershed(-D, markers, mask=self.image)
        mask1 = np.zeros((224, 224))
        mask1[self.labels!=0] = 255

        edged = cv2.Canny(np.uint8(mask1), 30, 200)
        self.contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if there is only one region we don't have to optimize the image
        if(len(self.contours)==1):
            # show the output image and save the respective outputs
            mask1 = np.zeros((224, 224), np.uint8)
            cv2.drawContours(mask1, self.contours, 0, 255, -1)
            cv2.imshow("Output", mask1)
            cv2.waitKey(0)
            cv2.imshow("GA", mask1)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_GA.jpg', mask1)
            cv2.imshow("AGA", mask1)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_AGA.jpg', mask1)
            cv2.imshow("PSO", mask1)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_PSO.jpg', mask1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        else:
            # if there are multiple regions in the image
            # make evry region isolated and with different values
            self.labels = np.float32(self.labels)
            for c in self.contours:
                mask = np.zeros((224, 224), np.uint8)
                cv2.drawContours(mask, [c], 0, 255, -1)
                mean_val = cv2.mean(self.labels,mask = mask)
                self.labels[mask == 255] = mean_val[0]
        # show the output image
        cv2.imshow("Output", mask1)
        cv2.waitKey(0)

        self.process_regions()

    #   processing image regions into respective values
    #   to proceed them with GA, AGA & PSO
    def process_regions(self):
        for i in np.unique(self.labels):
            if(i!=0):
                cords = np.where(self.labels == i)
                self.X.extend(list(cords[0]))
                self.Y.extend(list(cords[1]))
        self.rang = [i for i in range(len(self.X))]
        self.threshold = np.mean(np.unique(self.labels))
        self.GA()
        self.AGA()
        self.POS()
        cv2.destroyAllWindows()

    #   Creating population from the region values
    def populate(self):
        self.population = np.reshape(np.array(random.sample(self.rang, 300)), (30, 10))

    #   finding and saving the regions
    #   by evaluating the resluts of GA,AGA & PSO
    def produce_region(self, pixels, algo):
        lab = np.unique(self.labels)
        if(algo == 3):
            reg = self.eval_fitness([int(i) for i in pixels])
            if(reg != self.reg):
                reg = self.reg
            mask = np.zeros((224, 224))
            mask[self.labels == reg] = 255

        else:
            reg = [0]*len(lab)
            for i in pixels:
                reg[np.where(lab==self.gene_fitness(i))[0][0]] += 1

            mask = np.zeros((224, 224))
            self.reg = lab[reg.index(max(reg))]
            mask[self.labels == self.reg] = 255

        # show the output image
        if(algo == 1):
            cv2.imshow("GA", mask)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_GA.jpg', mask)
        elif(algo == 2):
            cv2.imshow("AGA", mask)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_AGA.jpg', mask)
        elif(algo == 3):
            cv2.imshow("PSO", mask)
            cv2.imwrite(self.output+self.path.split('/')[-1].split('.')[0]+'_PSO.jpg', mask)
        cv2.waitKey(0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #   Function to computes the fitness of the gene
    def gene_fitness(self, gene):
        return self.labels[self.X[gene]][self.Y[gene]]

    #   Function to compute the fitness of a chromosome
    def chromosome_fitness(self, chromosome):
        return sum([self.labels[self.X[gene]][self.Y[gene]] for gene in chromosome])

    #   Function that evaluate the fitness of the population
    def evaluate_fitness(self):
        self.fitnesses = []
        for chromosome in self.population:
            self.fitnesses.append(self.chromosome_fitness(chromosome))

    #   Function for roulette_selection from the population
    def roulette_selection(self):
        ind = random.randint(20, 29)
        self.chromosomes = self.population[:ind]

    #   Function for single_crossover of the parent_chromosomes
    def single_crossover(self):
        ind = random.randint(1, 9)
        for i in range(0, 20, 2):
            self.chromosomes[i], self.chromosomes[i+1] = np.append(self.chromosomes[i][:ind], self.chromosomes[i+1][ind:]), np.append(self.chromosomes[i+1][:ind], self.chromosomes[i][ind:])

    #   Function for fixed_mutation of the parent_chromosomes
    def fixed_mutation(self):
        for chromosome in self.chromosomes:
            fix = chromosome[self.fixed]
            new = chromosome[self.fixed]
            while(self.gene_fitness(fix) > self.gene_fitness(new) and new + 1 < len(self.X) and new not in chromosome):
                new += 1
            chromosome[self.fixed] = new

    #   Function for survival and update of the population from the children chromosomes
    def update_population(self):
        fit = []
        self.chromosomes = list(self.chromosomes)
        self.population = list(self.chromosomes)
        for i, chromosome in enumerate(self.chromosomes):
            if((self.population == chromosome).all(1).any() or self.chromosome_fitness(chromosome)<self.threshold*len(chromosome)):
                self.chromosomes.pop(i)
                continue
            fit.append(self.chromosome_fitness(chromosome))
        try:
            self.chromosomes = sorted(self.chromosomes, key = lambda chromo : fit)
            self.population.extend(self.chromosomes[:10])
        except:
            if(len(self.chromosomes)==0):
                self.iter = 600
                return
            self.population.extend(self.chromosomes)
        self.population = np.array(self.population)

    #   Genetic Algorithm for optimization of the image and furthur segmentation
    def GA(self):
        self.populate()
        while(self.iter<self.max_iter):
            self.evaluate_fitness()
            self.roulette_selection()
            self.single_crossover()
            self.fixed_mutation()
            self.update_population()
            self.iter+=1
        pixels = np.unique(np.array(self.population))
        self.produce_region(pixels, 1)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #   Function for cauchy_mutation of the parent_chromosomes
    def cauchy_mutation(self):
        for chromosome in self.chromosomes:
            mini = self.gene_fitness(chromosome[0])
            index = 0
            for ind, gene in enumerate(chromosome):
                if(self.gene_fitness(gene) > mini):
                    mini = self.gene_fitness(gene)
                    index = ind
            fix = chromosome[index]
            new = chromosome[index]
            while(self.gene_fitness(fix) > self.gene_fitness(new) and new + 1 < len(self.X) and new not in chromosome):
                new += 1
            chromosome[self.fixed] = new

    #   Function for random_mutation of the parent_chromosomes
    def random_mutation(self):
        for chromosome in self.chromosomes:
            index = random.randint(0, 9)
            fix = chromosome[index]
            new = chromosome[index]
            while(self.gene_fitness(fix) > self.gene_fitness(new) and new + 1 < len(self.X) and new not in chromosome):
                new += 1
            chromosome[self.fixed] = new

    #   Function for random_selection of the parent_chromosomes
    def selection(self):
        self.population = sorted(self.population, key=lambda pop:self.fitnesses)
        self.chromosomes = self.population[:20]

    #   Adaptive Genetic Algorithm for optimization of the image and furthur segmentation
    def AGA(self):
        self.populate()
        while(self.iter<self.max_iter):
            self.evaluate_fitness()
            self.selection()
            self.single_crossover()
            self.random_mutation()
            self.update_population()
            self.iter+=1
        pixels = np.unique(np.array(self.population))
        self.produce_region(pixels, 2)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #    Function to evaluate_fitness of the particles
    def eval_fitness(self, particle):
        return self.labels[particle[0]][particle[1]]

    #   Particle Swarm Optimization for optimization of the image and furthur segmentation
    def POS(self):
        initial=[120, 120]               # initial starting location [x1,x2...]
        bounds=[(0, 0),(223, 223)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
        pso = PSO(initial,bounds,num_particles=50,maxiter=20, labels=self.labels)
        final_particle = pso.process()
        self.produce_region(final_particle, 3)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


if(__name__=='__main__'):
    # construct the argument parse and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True, help="path to input image")
    #ap.add_argument("-o", "--output", required=True, help="path to output folder")
    #args = vars(ap.parse_args())
    path = input('Enter pic path: ') #"Benign/input/2.jpg"
    output = input('Enter output path: ') #"Benign/output/"
    img = Image(path, output)
    img.acquisition()

