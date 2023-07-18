# mastrangelo
Does dynamical sculpting happen? If so, how fast does it happen? Let's find out! 

Let's start from the beginning.

Dynamical sculpting is the idea that the orbital architectures of planetary systems excite over time, leading to configurations of higher mutual inclination and eccentricity. Side effects include planet-planet scattering that ejects planets from the system outright, and so sculpting can be imprinted in changes in the transit yield both through lower intrinsic counts and less favorable transit geometries. 

So, there's a time element, in order to build a population-wide picture of dynamical sculpting (among FGK dwarfs). We use Gaia-derived stellar ages from Berger+ 2020ab. And then we forward model from different putative sculpting timescales (eg. really fast sculpting over various timescales; really slow sculpting over various timescales) and see which models recover the observed Kepler transit multiplicity. 

Along the way, there are many potential "gotchas", and we try to address them systematically. 

Sample selection is done in the stars-and-planets.ipynb notebook in notebooks/. 

The workhorse functions for planetary system simulation, transit detection, and subsequent products like calculation of angular momentum deficit (AMD) and log likelihood compared to the observed Kepler transit multiplicity are in the mastrangelo/ folder. These functions are executed in embarrassingly parallel fashion on the University of Florida's HiPerGator HPC cluster.

Analysis notebooks will go under notebooks/. analyze.ipynb contains initial injection-recovery tests, figures for a conference showing fiducial transit multplicities to illustrate the "Kepler dichotomy", and comparison with the kinematic ages from the PAST collaboration. injection-recovery.ipynb contains injection-recovery experiments for a broader suite of representative models. 

#### A note on the name: 
[Bobbi Mastrangelo](https://bobbimastrangelo.com/) is a Floridian artist whose medium is manhole covers. Here is a manhole cover art that reminds me of planetary architectures ("When the People Care"): 

![Screen Shot 2023-04-26 at 3 23 54 PM](https://user-images.githubusercontent.com/16911363/234681422-eb24bdf5-9cba-4752-a35f-8da9ffa07a6f.png)

In particular, this sculpture makes me think of an alternate way that dynamical sculpting can imprint on a planetary system: changes in the period ratio between adjacent planets. That is, the spacing between orbits might change over time, rather than (or in addition to) the mutual inclinations and eccentricities.
