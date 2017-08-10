from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from numpy import dot, cross, arccos, sign, ceil
from numpy.linalg import norm

prmtop = AmberPrmtopFile('1ehz_single.prmtop')
inpcrd = AmberInpcrdFile('1ehz_single.inpcrd')

forcefield = app.ForceField('amber10.xml', 'amber10_obc.xml') #this is not used

system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolvent=OBC1)

pair_list = [(0,71),#start acceptor stem
			(1,70),
			(2,69),
			(3,68),
			(4,67),
			(5,66),
			(6,65), #end acceptor stem
			(9,24), #start D loop
			(10,23),
			(11,22),
			(12,21),#end D loop
			(26,42),#start anticodon stem
			(27,41),
			(28,40),
			(29,39),
			(30,38),#end anticodon stem
			(48,64),#start TC loop
			(49,63),
			(50,62),
			(51,61),
			(52,60)]#end TC loop

flush_stacking_pairs = [((6,65),(48,64))]
mismatch_stacking_pairs = [((9,24),(26,42))]

covariation_list = [(18,55),(14,47)] # linkages identified by covariation analysis

# then we create our own softcore potential (doi:10.1002/jcc.21806)
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

# now we add the base-base stacking term by building the energy function
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

base_pair_torsion_bond_list = []
coaxial_torsion_bond_list = []
base_pair_com_bond_list = []
coaxial_com_bond_list = []
covariation_com_bond_list = []


def restrain_backbone(res_index):
	print "called restrain backbone on the next residue:"
	print residue_list[res_index]
	res = residue_list[res_index]
	atoms = list(res.atoms())
	if res_index != 0: #if not the 5' terminal residue
		prev_res = residue_list[res_index-1]
		prev_atoms = list(prev_res.atoms())
		#alpha angle restraint
		a1 = next(a.index for a in prev_atoms if a.name=="O3'")
		a2 = next(a.index for a in atoms if a.name=="P")
		a3 = next(a.index for a in atoms if a.name=="O5'")
		a4 = next(a.index for a in atoms if a.name=="C5'")
		torsion_restraint_force.addTorsion(a1,a2,a3,a4,[0.0, -65*degrees]) #initial value of k = 0.0
		#beta angle restraint - nb we are lacking the 5' terminal phospahte!
		a1 = next(a.index for a in atoms if a.name=="P")
		a2 = next(a.index for a in atoms if a.name=="O5'")
		a3 = next(a.index for a in atoms if a.name=="C5'")
		a4 = next(a.index for a in atoms if a.name=="C4'")
		torsion_restraint_force.addTorsion(a1,a2,a3,a4,[0.0, 174*degrees]) #initial value of k = 0.0
	#gamma angle restraint
	a1 = next(a.index for a in atoms if a.name=="O5'")
	a2 = next(a.index for a in atoms if a.name=="C5'")
	a3 = next(a.index for a in atoms if a.name=="C4'")
	a4 = next(a.index for a in atoms if a.name=="C3'")
	torsion_restraint_force.addTorsion(a1,a2,a3,a4,[0.0, 54*degrees]) #initial value of k = 0.0
	if res_index != (len(residue_list)-1): #if not the 3' terminal residue
		next_res = residue_list[res_index+1]
		next_atoms = list(next_res.atoms())
		#epsilon angle restraint
		a1 = next(a.index for a in atoms if a.name=="C4'")
		a2 = next(a.index for a in atoms if a.name=="C3'")
		a3 = next(a.index for a in atoms if a.name=="O3'")
		a4 = next(a.index for a in next_atoms if a.name=="P")
		torsion_restraint_force.addTorsion(a1,a2,a3,a4,[0.0, -148*degrees]) #initial value of k = 0.0
		#zeta angle restraint
		a1 = next(a.index for a in atoms if a.name=="C3'")
		a2 = next(a.index for a in atoms if a.name=="O3'")
		a3 = next(a.index for a in next_atoms if a.name=="P")
		a4 = next(a.index for a in next_atoms if a.name=="O5'")
		torsion_restraint_force.addTorsion(a1,a2,a3,a4,[0.0, -71*degrees]) #initial value of k = 0.0

def makeAUbond(a_ind, u_ind):
	a_res = residue_list[a_ind]
	u_res = residue_list[u_ind]
	a1 = next(a.index for a in a_res.atoms() if a.name=="H61")
	a2 = next(a.index for a in u_res.atoms() if a.name=="O4")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in u_res.atoms() if a.name=="H3")
	a2 = next(a.index for a in a_res.atoms() if a.name=="N1")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0


def makeCGbond(c_ind, g_ind):
	c_res = residue_list[c_ind]
	g_res = residue_list[g_ind]
	a1 = next(a.index for a in c_res.atoms() if a.name=="H41")
	a2 = next(a.index for a in g_res.atoms() if a.name=="O6")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in g_res.atoms() if a.name=="H1")
	a2 = next(a.index for a in c_res.atoms() if a.name=="N3")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in g_res.atoms() if a.name=="H21")
	a2 = next(a.index for a in c_res.atoms() if a.name=="O2")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0

def makeGUbond(g_ind, u_ind):
	g_res = residue_list[g_ind]
	u_res = residue_list[u_ind]
	a1 = next(a.index for a in g_res.atoms() if a.name=="H1")
	a2 = next(a.index for a in u_res.atoms() if a.name=="O2")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in u_res.atoms() if a.name=="H3")
	a2 = next(a.index for a in g_res.atoms() if a.name=="O6")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0


def restrain_hbonds(pair):
	print "called restrain hbonds on the next 2 residues:"
	print residue_list[pair[0]]
	print residue_list[pair[1]]
	##really lame way to sort the base names to reduce duplication of code
	pair_dict = {residue_list[pair[0]].name[0]: pair[0],residue_list[pair[1]].name[0]:pair[1]}
	bases = sorted(pair_dict.keys())
	if bases == ['A','U']:
		makeAUbond(pair_dict['A'], pair_dict['U'])
	elif bases == ['C', 'G']:
		makeCGbond(pair_dict['C'], pair_dict['G'])
	elif bases == ['G', 'U']:
		makeGUbond(pair_dict['G'], pair_dict['U'])
	else:
		raise NotImplementedError("Can't handle this base pair")

def restrain_planar_torsions(pair, theta0):
	print "called restrain planar torsions on the next 2 residues:"
	print residue_list[pair[0]]
	print residue_list[pair[1]]
	print "with a theta0 value of", theta0
	indices = []
	for res_index in pair:
		res = residue_list[res_index]
		atoms = list(res.atoms())
		if res.name[0] in ['C', 'U']:
			indices.append(next(a.index for a in atoms if a.name=='C2'))
			indices.append(next(a.index for a in atoms if a.name=='C6'))
			indices.append(next(a.index for a in atoms if a.name=='C4'))
			indices.append(next(a.index for a in atoms if a.name=='N1'))
		elif res.name[0] in ['A', 'G']:
			indices.append(next(a.index for a in atoms if a.name=='N9'))
			indices.append(next(a.index for a in atoms if a.name=='N1'))
			indices.append(next(a.index for a in atoms if a.name=='N3'))
			indices.append(next(a.index for a in atoms if a.name=='N7'))
		else:
			raise NotImplementedError("torsions not implemented for base type", res.name[0])

	b_ind = base_stacking_force.addBond(indices, [0.0, theta0]) #initial value of k = 0.0
	return b_ind

def restrain_pair_com_distance(pair, next_pair, r0):
	print "called restrain pair COM on the next 4 residues:"
	print residue_list[pair[0]]
	print residue_list[pair[1]]
	print residue_list[next_pair[0]]
	print residue_list[next_pair[1]]
	print "with an r0 value of", r0
	g1 = []
	res = residue_list[pair[0]]
	atoms = list(res.atoms())
	g1.append(next(a.index for a in atoms if a.name=="C1'"))
	if "C8" in [a.name for a in atoms]:
		g1.append(next(a.index for a in atoms if a.name=="C8"))
	else:
		g1.append(next(a.index for a in atoms if a.name=="C6"))
	res = residue_list[pair[1]]
	print res
	atoms = list(res.atoms())
	g1.append(next(a.index for a in atoms if a.name=="C1'"))
	if "C8" in [a.name for a in atoms]:
		g1.append(next(a.index for a in atoms if a.name=="C8"))
	else:
		g1.append(next(a.index for a in atoms if a.name=="C6"))
	g2 = []
	res = residue_list[next_pair[0]]
	atoms = list(res.atoms())
	g2.append(next(a.index for a in atoms if a.name=="C1'"))
	if "C8" in [a.name for a in atoms]:
		g2.append(next(a.index for a in atoms if a.name=="C8"))
	else:
		g2.append(next(a.index for a in atoms if a.name=="C6"))
	res = residue_list[next_pair[1]]
	atoms = list(res.atoms())
	g2.append(next(a.index for a in atoms if a.name=="C1'"))
	if "C8" in [a.name for a in atoms]:
		g2.append(next(a.index for a in atoms if a.name=="C8"))
	else:
		g2.append(next(a.index for a in atoms if a.name=="C6"))

	assert(len(g1)==len(g2)==4)

	g1_ind = com_restraint_force.addGroup(g1)
	g2_ind = com_restraint_force.addGroup(g2)

	b_ind = com_restraint_force.addBond([g1_ind,g2_ind], [0.0, r0]) #initial value of k = 0.0
	return b_ind



def restrain_base_com_distance(pair, r0):
	print "called restrain base COM on the next 2 residues:"
	print residue_list[pair[0]]
	print residue_list[pair[1]]
	print "with an r0 value of", r0

	res = residue_list[pair[0]]
	atoms = list(res.atoms())
	g1 = []
	for a in atoms:
		if a.name[-1] != "'" and "P" not in a.name:
			g1.append(a.index)

	res = residue_list[pair[1]]
	atoms = list(res.atoms())
	g2 = []
	for a in atoms:
		if a.name[-1] != "'" and "P" not in a.name:
			g2.append(a.index)

	g1_ind = com_restraint_force.addGroup(g1)
	g2_ind = com_restraint_force.addGroup(g2)

	b_ind = com_restraint_force.addBond([g1_ind,g2_ind], [0.0, r0]) #initial value of k = 0.0
	return b_ind

##get list of all residues
residue_list = list(prmtop.topology.residues())

##the actual restraint algorithm
for pair in pair_list:
	restrain_backbone(pair[0])
	restrain_backbone(pair[1])
	restrain_hbonds(pair)
	restrain_planar_torsions(pair, 160*degrees)

	next_pair = (pair[0]+1,pair[1]-1)
	if next_pair in pair_list:
		com_b_ind = restrain_pair_com_distance(pair, next_pair, 4.5*angstrom)
		torsion1_b_ind = restrain_planar_torsions((pair[0],next_pair[1]), 160*degrees)
		torsion2_b_ind = restrain_planar_torsions((pair[1],next_pair[0]), 160*degrees)
		base_pair_com_bond_list.append(com_b_ind)
		base_pair_torsion_bond_list.append(torsion1_b_ind)
		base_pair_torsion_bond_list.append(torsion2_b_ind)

for quad in flush_stacking_pairs:
	pair1 = quad[0]
	pair2 = quad[1]
	com_b_ind = restrain_pair_com_distance(pair1, pair2, 4.5*angstrom)
	torsion1_b_ind = restrain_planar_torsions((pair1[0],pair2[1]), 160*degrees)
	torsion2_b_ind = restrain_planar_torsions((pair1[1],pair2[0]), 160*degrees)
	coaxial_com_bond_list.append(com_b_ind)
	coaxial_torsion_bond_list.append(torsion1_b_ind)
	coaxial_torsion_bond_list.append(torsion2_b_ind)

for quad in mismatch_stacking_pairs:
	pair1 = quad[0]
	pair2 = quad[1]
	com_b_ind = restrain_pair_com_distance(pair1, pair2, 12*angstrom)
	torsion1_b_ind = restrain_planar_torsions((pair1[0],pair2[1]), 150*degrees)
	torsion2_b_ind = restrain_planar_torsions((pair1[1],pair2[0]), 150*degrees)
	coaxial_com_bond_list.append(com_b_ind)
	coaxial_torsion_bond_list.append(torsion1_b_ind)
	coaxial_torsion_bond_list.append(torsion2_b_ind)


for pair in covariation_list:
	com_b_ind = restrain_base_com_distance(pair, 7.5*angstrom)
	covariation_com_bond_list.append(com_b_ind)



charges = []
exception_charge_prods = []
openmm_nonbonded_force = []
openmm_solvent_force = []
for (i, f) in reversed(list(enumerate(system.getForces()))):
	print str(i) + ' - ' + str(f)
	if isinstance(f, openmm.CustomGBForce):
		openmm_solvent_force = f
	elif isinstance(f, openmm.NonbondedForce):
		openmm_nonbonded_force = f
		for atom in range(f.getNumParticles()):
			(charge,sigma,epsilon) = f.getParticleParameters(atom)
			softcore_force.addParticle([sigma, epsilon, 0.0]) # initial value of lamb = 0.0
		bondlist = [(b[0].index, b[1].index) for b in prmtop.topology.bonds()]
		softcore_force.createExclusionsFromBonds(bondlist, 3)

		
system.addForce(base_stacking_force)
system.addForce(torsion_restraint_force)
system.addForce(hbond_restraint_force)
system.addForce(com_restraint_force)
system.addForce(softcore_force)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
opencl_platform = []
for p in range(openmm.Platform.getNumPlatforms()):
	if openmm.Platform.getPlatform(p).getName() == "OpenCL":
		opencl_platform = openmm.Platform.getPlatform(p)

simulation = Simulation(prmtop.topology, system, integrator, platform=opencl_platform)
simulation.context.setPositions(inpcrd.positions)

# we set charges to 0.0 after initializing the system, otherwise OpenMM gets a little
# clever and decides not to include the coulomb potential
charges = []
exception_charge_prods = []
exception_sigmas = []
exception_epsilons = []
solvent_charges = []
for a_ind in range(openmm_nonbonded_force.getNumParticles()):
	(charge,sigma,epsilon) = openmm_nonbonded_force.getParticleParameters(a_ind)
	charges.append(charge)
	openmm_nonbonded_force.setParticleParameters(a_ind, 0.0, 0.0, 0.0) # no charge, no sigma, no epsilon
openmm_nonbonded_force.updateParametersInContext(simulation.context)

for e_ind in range(openmm_nonbonded_force.getNumExceptions()):
	(a1, a2, chargeprod, sigma, epsilon) = openmm_nonbonded_force.getExceptionParameters(e_ind)
	exception_charge_prods.append(chargeprod)
	exception_sigmas.append(sigma)
	exception_epsilons.append(epsilon)
	openmm_nonbonded_force.setExceptionParameters(e_ind, a1, a2, 0.0, 0.0, 0.0)
openmm_nonbonded_force.updateParametersInContext(simulation.context)

for a_ind in range(openmm_solvent_force.getNumParticles()):
	(q, o_r, s_r) = openmm_solvent_force.getParticleParameters(a_ind)
	solvent_charges.append(q)
	openmm_solvent_force.setParticleParameters(a_ind, [0.0, o_r, s_r])
openmm_solvent_force.updateParametersInContext(simulation.context)


pdb_reporter = PDBReporter('output.pdb', 1000)
state_reporter = StateDataReporter(stdout, 1000, step=True, speed=True,
           potentialEnergy=True, temperature=True)


pdb_reporter.report(simulation, simulation.context.getState(getPositions=True))

simulation.reporters.append(pdb_reporter)
simulation.reporters.append(state_reporter)

simulation.minimizeEnergy()

print "PHASE 1: WARMING - 0.25 ns"
for i in range(250):
	integrator.setTemperature((300+2.8*(i+1))*kelvin)
	simulation.step(1000)
    
print "PHASE 2: CLOSE BASE PAIR RESTRAINTS - 1 ns"
# ##those within 40 nucleotides (approximately 1300 atoms)
for i in range(1000):
	# base stacking force
	for bond_ind in base_pair_torsion_bond_list:
		[atoms,params] = base_stacking_force.getBondParameters(bond_ind)
		[k, theta0] = params
		if (max(atoms)-min(atoms) <= 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole # will end at k = 10, as desired
		base_stacking_force.setBondParameters(bond_ind, atoms, [k, theta0])
	base_stacking_force.updateParametersInContext(simulation.context)

	# torsion restraints
	for t_ind in range(torsion_restraint_force.getNumTorsions()):
		[a1, a2, a3, a4, params] = torsion_restraint_force.getTorsionParameters(t_ind)
		[k, theta0] = params
		k = 0.01*(i+1)*kilocalorie_per_mole # will end at k = 10, as desired
		torsion_restraint_force.setTorsionParameters(t_ind, a1, a2, a3, a4, [k, theta0])
	torsion_restraint_force.updateParametersInContext(simulation.context)

	# hydrogen bond restraints
	for b_ind in range(hbond_restraint_force.getNumBonds()):
		[a1, a2, params] = hbond_restraint_force.getBondParameters(b_ind)
		[k] = params
		if (max([a1,a2]) - min([a1,a2]) <= 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		hbond_restraint_force.setBondParameters(b_ind, a1, a2, [k])
	hbond_restraint_force.updateParametersInContext(simulation.context)

	# center of mass restraints
	for b_ind in base_pair_com_bond_list:
		[groups, params] = com_restraint_force.getBondParameters(b_ind)
		[k, r0] = params
		bonded_atoms = []
		for g_ind in groups:
			[atoms, weights] = com_restraint_force.getGroupParameters(g_ind)
			bonded_atoms.extend(atoms)
		if (max(bonded_atoms) - min(bonded_atoms) <= 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		com_restraint_force.setBondParameters(b_ind, groups, [k, r0])
	com_restraint_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 2.5: FAR BASE PAIR RESTRAINTS - 1 ns"
## for those beyond 40 nucleotides (approximately 1300 atoms)
for i in range(1000):
	# base stacking force
	for bond_ind in base_pair_torsion_bond_list:
		[atoms,params] = base_stacking_force.getBondParameters(bond_ind)
		[k, theta0] = params
		if (max(atoms)-min(atoms) > 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole # will end at k = 10, as desired
		base_stacking_force.setBondParameters(bond_ind, atoms, [k, theta0])
	base_stacking_force.updateParametersInContext(simulation.context)

	# hydrogen bond restraints
	for b_ind in range(hbond_restraint_force.getNumBonds()):
		[a1, a2, params] = hbond_restraint_force.getBondParameters(b_ind)
		[k] = params
		if (max([a1,a2]) - min([a1,a2]) > 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		hbond_restraint_force.setBondParameters(b_ind, a1, a2, [k])
	hbond_restraint_force.updateParametersInContext(simulation.context)

	# center of mass restraints
	for b_ind in base_pair_com_bond_list:
		[groups, params] = com_restraint_force.getBondParameters(b_ind)
		[k, r0] = params
		bonded_atoms = []
		for g_ind in groups:
			[atoms, weights] = com_restraint_force.getGroupParameters(g_ind)
			bonded_atoms.extend(atoms)
		if (max(bonded_atoms) - min(bonded_atoms) > 1300):
			k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		com_restraint_force.setBondParameters(b_ind, groups, [k, r0])
	com_restraint_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 3: REINTRODUCTION OF VAN DER WAALS FORCES - 0.25 ns"
## lambda is soft-core potential scaling factor
for i in range(250):
	for a_ind in range(softcore_force.getNumParticles()):
		[sigma, epsilon, lamb] = softcore_force.getParticleParameters(a_ind)
		lamb = 0.004*(i+1)
		softcore_force.setParticleParameters(a_ind, [sigma, epsilon, lamb])
	softcore_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 3.5: REINTRODUCTION OF 1-4 SCALED VAN DER WAALS FORCES - 0.25 ns"
## lambda is soft-core potential scaling factor
for i in range(250):
	for e_ind in range(openmm_nonbonded_force.getNumExceptions()):
		[a1, a2, chargeprod, sigma, epsilon] = openmm_nonbonded_force.getExceptionParameters(e_ind)
		sigma = exception_sigmas[e_ind]*(0.004*(i+1))
		epsilon = exception_epsilons[e_ind]*(0.004*(i+1))
		openmm_nonbonded_force.setExceptionParameters(e_ind, a1, a2, chargeprod, sigma, epsilon)
	openmm_nonbonded_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 4: REINTRODUCTION OF COULOMB FORCES - 0.25 ns"
# I will be scaling the charges linearly from 0 to 1 * their original value
for i in range(250):
	for a_ind in range(openmm_nonbonded_force.getNumParticles()):
		[charge, sigma, epsilon] = openmm_nonbonded_force.getParticleParameters(a_ind)
		#we use the square root of our scaling function since charges are multiplied together
		charge = charges[a_ind]*sqrt(0.004*(i+1))
		openmm_nonbonded_force.setParticleParameters(a_ind, charge, sigma, epsilon)

	#openmm also has the exceptions list
	for e_ind in range(openmm_nonbonded_force.getNumExceptions()):
		[a1, a2, chargeprod, sigma, epsilon] = openmm_nonbonded_force.getExceptionParameters(e_ind)
		#in this case we are using pre-multiplied values, so no square root
		chargeprod = exception_charge_prods[e_ind]*(0.004*(i+1))
		openmm_nonbonded_force.setExceptionParameters(e_ind, a1, a2, chargeprod, sigma, epsilon)

	openmm_nonbonded_force.updateParametersInContext(simulation.context)

	for a_ind in range(openmm_solvent_force.getNumParticles()):
		(q, o_r, s_r) = openmm_solvent_force.getParticleParameters(a_ind)
		charge = solvent_charges[a_ind]*sqrt(0.004*(i+1))
		openmm_solvent_force.setParticleParameters(a_ind, [charge, o_r, s_r])
	openmm_solvent_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 5: HELICAL STACKING AND TERTIARY INTERACTIONS - 1 ns"
for i in range(1000):
	# base stacking force
	for b_ind in coaxial_torsion_bond_list:
		[atoms,params] = base_stacking_force.getBondParameters(b_ind)
		[k, theta0] = params
		k = 0.01*(i+1)*kilocalorie_per_mole # will end at k = 10, as desired
		base_stacking_force.setBondParameters(b_ind, atoms, [k, theta0])
	base_stacking_force.updateParametersInContext(simulation.context)

	# center of mass restraints
	for b_ind in coaxial_com_bond_list:
		[groups, params] = com_restraint_force.getBondParameters(b_ind)
		[k, r0] = params
		k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		com_restraint_force.setBondParameters(b_ind, groups, [k, r0])

	for b_ind in covariation_com_bond_list:
		[groups, params] = com_restraint_force.getBondParameters(b_ind)
		[k, r0] = params
		k = 0.01*(i+1)*kilocalorie_per_mole/(angstrom**2) # will end at k = 10, as desired
		com_restraint_force.setBondParameters(b_ind, groups, [k, r0])
	com_restraint_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 6: COOLING - 2.5 ns"
## here we do the actual annealing
for i in range(1750):
	integrator.setTemperature((1000-(0.4*(i+1)))*kelvin)
	simulation.step(1000)

integrator.setStepSize(0.002*picoseconds)
pdb_reporter._reportInterval = 500
state_reporter._reportInterval = 500
for i in range(750):
	integrator.setTemperature((300-(0.4*(i+1)))*kelvin)
	simulation.step(500)

#not sure if required? PHASE 7
simulation.minimizeEnergy()
pdb_reporter.report(simulation, simulation.context.getState(getPositions=True))
state = simulation.context.getState(getEnergy=True, groups={31})
restraint_energy = state.getPotentialEnergy()
print "Final restraint energy:", restraint_energy