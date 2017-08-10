from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
# here we create our softcore potential (doi:10.1002/jcc.21806)
terms = []
terms.append("4*epsilon*lamb*(frac^2-frac)")
terms.append("frac=(1/(alpha*(1-lamb)+(r/sigma)^6))")
terms.append("sigma=(0.5*(sigma1+sigma2))")
terms.append("epsilon=sqrt(epsilon1*epsilon2)")
terms.append("lamb=(0.5*(lamb1+lamb2))")
softcore_energy_function = '; '.join(terms)

softcore_force = openmm.CustomNonbondedForce(softcore_energy_function)
softcore_force.addPerParticleParameter("sigma")
softcore_force.addPerParticleParameter("epsilon")
softcore_force.addPerParticleParameter("lamb") # scaling factor, intially 0.0
softcore_force.addGlobalParameter("alpha", 0.5)


# here we add the base-base stacking term by building the energy function
# some abbreviations used: theta = angle
#                             dp = dot product
#                            ncp = normalized cross product
#                             cp = cross product
#                             nf = normalizing factor (aka magnitude)
#                             dv = displacement vector
terms = []
terms.append("k*step(-(theta-theta0))*(theta-theta0)^2")
terms.append("theta=acos(dp*0.999)") # hack to avoid edge case of dp = 1.0
terms.append("dp=ncp1_x*ncp2_x+ncp1_y*ncp2_y+ncp1_z*ncp2_z")
terms.append("dp=cp1_x*cp2_x+cp1_y*cp2_y+cp1_z*cp2_z")
terms.append("ncp1_x=cp1_x/nf1")
terms.append("ncp1_y=cp1_y/nf1")
terms.append("ncp1_z=cp1_z/nf1")
terms.append("ncp2_x=cp2_x/nf2")
terms.append("ncp2_y=cp2_y/nf2")
terms.append("ncp2_z=cp2_z/nf2")
terms.append("nf1=sqrt(cp1_x^2+cp1_y^2+cp1_z^2)")
terms.append("nf2=sqrt(cp2_x^2+cp2_y^2+cp2_z^2)")
terms.append("cp1_x=dv1_y*dv2_z-dv1_z*dv2_y")
terms.append("cp1_y=dv1_z*dv2_x-dv1_x*dv2_z")
terms.append("cp1_z=dv1_x*dv2_y-dv1_y*dv2_x")
terms.append("cp2_x=dv3_y*dv4_z-dv3_z*dv4_y")
terms.append("cp2_y=dv3_z*dv4_x-dv3_x*dv4_z")
terms.append("cp2_z=dv3_x*dv4_y-dv3_y*dv4_x")
terms.append("dv1_x=x1-x2")
terms.append("dv1_y=y1-y2")
terms.append("dv1_z=z1-z2")
terms.append("dv2_x=x3-x4")
terms.append("dv2_y=y3-y4")
terms.append("dv2_z=z3-z4")
terms.append("dv3_x=x5-x6")
terms.append("dv3_y=y5-y6")
terms.append("dv3_z=z5-z6")
terms.append("dv4_x=x7-x8")
terms.append("dv4_y=y7-y8")
terms.append("dv4_z=z7-z8")
base_stacking_energy_function = '; '.join(terms)

base_stacking_force = openmm.CustomCompoundBondForce(8,base_stacking_energy_function)
base_stacking_force.addPerBondParameter('k')
base_stacking_force.addPerBondParameter('theta0')
base_stacking_force.setForceGroup(31)


##implements a flat bottom harmonic well centered on theta0 +/- w
##used in backbone torsions
terms = []
terms.append("k*(step(-(thetap-(theta0-w)))*(thetap-(theta0-w))^2+step(thetap-(theta0+w))*(thetap-(theta0+w))^2)")
terms.append("thetap=step(-((theta-theta0)+pi))*2*pi+theta+step((theta-theta0)-pi)*(-2*pi)")
torsion_restraint_energy_function = '; '.join(terms)

torsion_restraint_force = openmm.CustomTorsionForce(torsion_restraint_energy_function)
torsion_restraint_force.addGlobalParameter("w", 15*degrees)
torsion_restraint_force.addGlobalParameter("pi", 3.14159)
torsion_restraint_force.addPerTorsionParameter("k")
torsion_restraint_force.addPerTorsionParameter("theta0")
torsion_restraint_force.setForceGroup(31)


##implements a flat bottom harmonic distance well with specified min and max r
##used in H-bonding
hbond_restraint_force = openmm.CustomBondForce("k*(step(-(r-min))*(r-min)^2+step(r-max)*(r-max)^2)")
hbond_restraint_force.addPerBondParameter("k")
hbond_restraint_force.addGlobalParameter("min", 1.6*angstrom)
hbond_restraint_force.addGlobalParameter("max", 2.0*angstrom)
hbond_restraint_force.setForceGroup(31)


##implments Center Of Mass distance harmonic flat-bottom well
##used in helical base-pair stacking as well as covariance and biochemical restraints
terms = []
terms.append("k*step(r-r0)*(r-r0)^2")
terms.append("r=sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2+0.001)") #hack to avoid edge case of r=0
com_energy_function = '; '.join(terms)

com_restraint_force = openmm.CustomCentroidBondForce(2, com_energy_function)
com_restraint_force.addPerBondParameter("k")
com_restraint_force.addPerBondParameter("r0")
com_restraint_force.setForceGroup(31)

#Force Group is set to 31 because we want to be able to specifically print
# the restraint energy later on.  this has no effect on the simulation, it's 
# strictly a form of labelling