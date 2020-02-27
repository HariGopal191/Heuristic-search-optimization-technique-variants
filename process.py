# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:47:58 2020

@author: HariGopal V
"""

import cv2
import numpy as np
from skimage.morphology import watershed
from scipy import ndimage
from skimage.feature import peak_local_max
import argparse
import imutils
import random


class Image:

    def __init__(self, path):
        self.path = path
        self.image = None
        self.gray = None
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

    # reading/aquiring the scan image
    def acquisition(self):
        self.image = cv2.imread(self.path)
        self.Pre_Processing()

    def thresholding(self):
        return 90

    # Otsu's thresholding after mean shift filtering to0 aid thrtesholding
    def Pre_Processing(self):
        self.image = cv2.resize(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (224, 224), interpolation = cv2.INTER_NEAREST)
        self.gray = self.image
        thresh = self.thresholding()
        th3, self.image = cv2.threshold(self.image,thresh,255,cv2.THRESH_TOZERO)
        # show the output image
        cv2.imshow("thresh", self.image)
        cv2.waitKey(0)
        self.De_Noising()

    # MedianBlur denoising for removing of salt-pepper noise
    def De_Noising(self):
        self.image = cv2.medianBlur(np.float32(self.image), 5)
        # show the output image
        cv2.imshow("Blur", self.image)
        cv2.waitKey(0)

        self.segment()

    # watershed segmentation
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
        if(len(self.contours)==1):
            # show the output image
            cv2.imshow("Output", mask1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        else:
            self.labels = np.float32(self.labels)
            for c in self.contours:
                mask = np.zeros((224, 224), np.uint8)
                cv2.drawContours(mask, [c], 0, 255, -1)
                #lab = np.unique(self.labels[mask == 255])
                mean_val = cv2.mean(self.labels,mask = mask)
                self.labels[mask == 255] = mean_val[0]
        # show the output image
        cv2.imshow("Output", mask1)
        cv2.waitKey(0)

        self.process_regions()


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


    def populate(self):
        self.population = np.reshape(np.array(random.sample(self.rang, 300)), (30, 10))


    def produce_region(self, pixels, algo):
        lab = np.unique(self.labels)
        reg = [0]*len(lab)
        for i in pixels:
            reg[np.where(lab==self.gene_fitness(i))[0][0]] += 1

        mask = np.zeros((224, 224))
        mask[self.labels == lab[reg.index(max(reg))]] = 255
        # show the output image
        if(algo == 1):
            cv2.imshow("GA", mask)
            cv2.imwrite("Benign/output/"+self.path.split('/')[-1].split('.')[0]+'_GA.jpg', mask)
        elif(algo == 2):
            cv2.imshow("AGA", mask)
            cv2.imwrite("Benign/output/"+self.path.split('/')[-1].split('.')[0]+'_AGA.jpg', mask)
        cv2.waitKey(0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def gene_fitness(self, gene):
        return self.labels[self.X[gene]][self.Y[gene]]

    def chromosome_fitness(self, chromosome):
        return sum([self.labels[self.X[gene]][self.Y[gene]] for gene in chromosome])

    def evaluate_fitness(self):
        self.fitnesses = []
        for chromosome in self.population:
            self.fitnesses.append(self.chromosome_fitness(chromosome))

    def roulette_selection(self):
        ind = random.randint(20, 29)
        self.chromosomes = self.population[:ind]

    def single_crossover(self):
        ind = random.randint(1, 9)
        for i in range(0, 20, 2):
            self.chromosomes[i], self.chromosomes[i+1] = np.append(self.chromosomes[i][:ind], self.chromosomes[i+1][ind:]), np.append(self.chromosomes[i+1][:ind], self.chromosomes[i][ind:])

    def fixed_mutation(self):
        for chromosome in self.chromosomes:
            fix = chromosome[self.fixed]
            new = chromosome[self.fixed]
            while(self.gene_fitness(fix) > self.gene_fitness(new) and new + 1 < len(self.X) and new not in chromosome):
                new += 1
            chromosome[self.fixed] = new


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

    def random_mutation(self):
        for chromosome in self.chromosomes:
            index = random.randint(0, 9)
            fix = chromosome[index]
            new = chromosome[index]
            while(self.gene_fitness(fix) > self.gene_fitness(new) and new + 1 < len(self.X) and new not in chromosome):
                new += 1
            chromosome[self.fixed] = new


    def selection(self):
        self.population = sorted(self.population, key=lambda pop:self.fitnesses)
        self.chromosomes = self.population[:20]


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


    def POS(self):
        pass


if(__name__=='__main__'):
    # construct the argument parse and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True, help="path to input image")
    #args = vars(ap.parse_args())
    path = 'Benign/input/2.jpg' #input('Enter pic path: ')
    img = Image(path)
    img.acquisition()
    labels = img.labels
    pop = img.population
    fit = img.fitnesses
    chrome = img.chromosomes
    itera = img.iter
    contours = img.contours
    X = img.X
    Y = img.Y
