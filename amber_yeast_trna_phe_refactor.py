from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from numpy import dot, cross, arccos, sign, ceil
from numpy.linalg import norm
from custom_forces import *

#load input files
prmtop = AmberPrmtopFile('1ehz_single.prmtop')
inpcrd = AmberInpcrdFile('1ehz_single.inpcrd')

#creates the default AMBER force field
forcefield = app.ForceField('amber10.xml', 'amber10_obc.xml') #this is not used

system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolvent=OBC1)

###Begin Secondary Structure Specification
#the pair list contains tuples with paired bases.  no repetition is allowed and the base
#with the lower index (i.e. closert to 5' end) comes first
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

#pairs of pairs used to define helical stacking interactions.  again, the 5' end
#pair comes first
flush_stacking_pairs = [((6,65),(48,64))]
mismatch_stacking_pairs = [((9,24),(26,42))]

# linkages identified by covariation or biochemical analysis
covariation_list = [(18,55),(14,47)] 

# linkages identified by MOHCA (not tested, not used)
mohca_list = []

###End Secondary Structure Specification
###Begin Restraint Specification
#note that all restraints are intitially set up with k=0 (i.e. disabled)
#the value of k will be increased to 10 during the simulation

#lists that are used internally to track bond indices
base_pair_torsion_bond_list = []
coaxial_torsion_bond_list = []
base_pair_com_bond_list = []
coaxial_com_bond_list = []
covariation_com_bond_list = []
mohca_bond_list = []

#restrains the phosphosugar backbone for a single nucleotide with index res_index
def restrain_backbone(res_index):
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

#creates HBond restraints for A-U canonical base pair
def makeAUbond(a_ind, u_ind):
	a_res = residue_list[a_ind]
	u_res = residue_list[u_ind]
	a1 = next(a.index for a in a_res.atoms() if a.name=="H61")
	a2 = next(a.index for a in u_res.atoms() if a.name=="O4")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in u_res.atoms() if a.name=="H3")
	a2 = next(a.index for a in a_res.atoms() if a.name=="N1")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0

#creates HBond restraints for G-C canonical base pair
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

#creates HBond restraints for G-U noncanonical wobble base pair
def makeGUbond(g_ind, u_ind):
	g_res = residue_list[g_ind]
	u_res = residue_list[u_ind]
	a1 = next(a.index for a in g_res.atoms() if a.name=="H1")
	a2 = next(a.index for a in u_res.atoms() if a.name=="O2")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0
	a1 = next(a.index for a in u_res.atoms() if a.name=="H3")
	a2 = next(a.index for a in g_res.atoms() if a.name=="O6")
	hbond_restraint_force.addBond(a1,a2, [0.0]) #initial value of k = 0.0


#calls the appropriate HBond function on each base pair
def restrain_hbonds(pair):
	##sort the base names alphabetically to reduce duplication of code
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

#restrains the planes of two bases on OPPOSITE sides of a helix
#the base planes should always be roughly antiparallel (180 degrees)
def restrain_planar_torsions(pair, theta0):
	indices = []
	for res_index in pair:
		res = residue_list[res_index]
		atoms = list(res.atoms())
		#which atoms are affected depends on base type
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
	return b_ind # the bond index is returned to keep track of later

#restrains the center of mass between base pairs, as defined by seetin & mathews
# (i.e. not all atoms are used).  atoms must be added into 2 different groups,
# and then the groups' centers of mass are restrained
def restrain_pair_com_distance(pair, next_pair, r0):
	g1 = []
	res = residue_list[pair[0]]
	atoms = list(res.atoms())
	g1.append(next(a.index for a in atoms if a.name=="C1'"))
	if "C8" in [a.name for a in atoms]:
		g1.append(next(a.index for a in atoms if a.name=="C8"))
	else:
		g1.append(next(a.index for a in atoms if a.name=="C6"))
	res = residue_list[pair[1]]

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

	assert(len(g1)==len(g2)==4) #sanity check

	g1_ind = com_restraint_force.addGroup(g1)
	g2_ind = com_restraint_force.addGroup(g2)

	b_ind = com_restraint_force.addBond([g1_ind,g2_ind], [0.0, r0]) #initial value of k = 0.0
	return b_ind # the bond index is returned to keep track of later

#here we restrain individual BASE (not pair!) centers of mass, used for the biochemical
# or covariation restraints.  I defined the base as "everything that doesn't have prime
# in the name".  Also since the P doesn't have prime, it needs to be excluded too.
def restrain_base_com_distance(pair, r0):
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
	return b_ind # the bond index is returned to keep track of later

#here we restrain the C4' atoms for MOHCA by re-using center of mass restraint code
#the center of mass of one atom is just the position of that atom.  We don't use a normal
#bond force because we still need the flat-bottomed well
# NOT USED, NOT TESTED!
def restrain_c4_distance(pair, r0):
	res = residue_list[pair[0]]
	atoms = list(res.atoms())
	g1 = []
	for a in atoms:
		if a.name == "C4'":
			g1.append(a.index)
	res = residue_list[pair[1]]
	atoms = list(res.atoms())
	g2 = []
	for a in atoms:
		if a.name == "C4'":
			g2.append(a.index)

	g1_ind = com_restraint_force.addGroup(g1)
	g2_ind = com_restraint_force.addGroup(g2)

	b_ind = com_restraint_force.addBond([g1_ind,g2_ind], [0.0, r0]) #initial value of k = 0.0
	return b_ind # the bond index is returned to keep track of later

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

#not tested!
for pair in mohca_list:
	mohca_b_ind = restrain_c4_distance(pair, 25*angstrom)
	mohca_bond_list.append(com_b_ind)

###End Restraint Specification

#The following section deals with copying the parameters from OpenMM to our
#softcore potential.  We also save references to some default OpenMM classes
#that we will need to modify later
openmm_nonbonded_force = []
openmm_solvent_force = []
for f in system.getForces():
	if isinstance(f, openmm.CustomGBForce):
		openmm_solvent_force = f
	elif isinstance(f, openmm.NonbondedForce):
		openmm_nonbonded_force = f
		for atom in range(f.getNumParticles()):
			(charge,sigma,epsilon) = f.getParticleParameters(atom)
			softcore_force.addParticle([sigma, epsilon, 0.0]) # initial value of lamb = 0.0

#here we set up exceptions, which disables our softcore potential on
#atoms that are within 3 bonds of each other
bondlist = [(b[0].index, b[1].index) for b in prmtop.topology.bonds()]
softcore_force.createExclusionsFromBonds(bondlist, 3)

#add the actual forces to the system, now that they are prepared
system.addForce(base_stacking_force)
system.addForce(torsion_restraint_force)
system.addForce(hbond_restraint_force)
system.addForce(com_restraint_force)
system.addForce(softcore_force)

#create the integrator and set up the simulation with default coordinates
#initial temperature is 300K
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# some lists to keep track of original parameters.  we're going to set them to 0
# in the context and then slowly bring them back to the values in these lists
charges = []
exception_charge_prods = []
exception_sigmas = []
exception_epsilons = []
solvent_charges = []

# we set charges to 0.0 AFTER initializing the system, otherwise OpenMM gets a little
# clever and decides not to include the coulomb potential at all!
#disable built-in L-J and coulomb forces by setting parameters to 0
for a_ind in range(openmm_nonbonded_force.getNumParticles()):
	(charge,sigma,epsilon) = openmm_nonbonded_force.getParticleParameters(a_ind)
	charges.append(charge)
	openmm_nonbonded_force.setParticleParameters(a_ind, 0.0, 0.0, 0.0) # no charge, no sigma, no epsilon
openmm_nonbonded_force.updateParametersInContext(simulation.context)

#need to do again for pre-computed 1-4 interaction parameters
for e_ind in range(openmm_nonbonded_force.getNumExceptions()):
	(a1, a2, chargeprod, sigma, epsilon) = openmm_nonbonded_force.getExceptionParameters(e_ind)
	exception_charge_prods.append(chargeprod)
	exception_sigmas.append(sigma)
	exception_epsilons.append(epsilon)
	openmm_nonbonded_force.setExceptionParameters(e_ind, a1, a2, 0.0, 0.0, 0.0)
openmm_nonbonded_force.updateParametersInContext(simulation.context)

#and again for the solvent force
for a_ind in range(openmm_solvent_force.getNumParticles()):
	(q, o_r, s_r) = openmm_solvent_force.getParticleParameters(a_ind)
	solvent_charges.append(q)
	openmm_solvent_force.setParticleParameters(a_ind, [0.0, o_r, s_r])
openmm_solvent_force.updateParametersInContext(simulation.context)

# create reporters for output
pdb_reporter = PDBReporter('output.pdb', 1000)
state_reporter = StateDataReporter(stdout, 1000, step=True, speed=True,
           potentialEnergy=True, temperature=True)

#report the initial structure.  helps VMD detect bonds properly
pdb_reporter.report(simulation, simulation.context.getState(getPositions=True))

#add reporters to simulation
simulation.reporters.append(pdb_reporter)
simulation.reporters.append(state_reporter)

#minimize energy
simulation.minimizeEnergy()

###Begin Annealing Protocol
print "PHASE 1: WARMING - 0.25 ns"
#final temperature is 1000K
for i in range(250):
	integrator.setTemperature((300+2.8*(i+1))*kelvin)
	simulation.step(1000)
    
print "PHASE 2: CLOSE BASE PAIR RESTRAINTS - 1 ns"
##those within 40 nucleotides (approximately 1300 atoms) are affected first
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
# since lambda is already at 1, the softcore already equals regular L-J,
# this just adds back the missing 1-4 interactions
for i in range(250):
	for e_ind in range(openmm_nonbonded_force.getNumExceptions()):
		[a1, a2, chargeprod, sigma, epsilon] = openmm_nonbonded_force.getExceptionParameters(e_ind)
		sigma = exception_sigmas[e_ind]*(0.004*(i+1))
		epsilon = exception_epsilons[e_ind]*(0.004*(i+1))
		openmm_nonbonded_force.setExceptionParameters(e_ind, a1, a2, chargeprod, sigma, epsilon)
	openmm_nonbonded_force.updateParametersInContext(simulation.context)

	simulation.step(1000)

print "PHASE 4: REINTRODUCTION OF COULOMB FORCES - 0.25 ns"
# I will be scaling the charges linearly from 0 to their original value
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

	# MOHCA restraints (not used, not tested)
	for b_ind in mohca_bond_list:
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

#as per seeting & mathews, at the 300K mark we icrease the timestep
integrator.setStepSize(0.002*picoseconds)
#reporters are adjusted too so that 1 frame = 1 ps
pdb_reporter._reportInterval = 500
state_reporter._reportInterval = 500

#continue annealing
for i in range(750):
	integrator.setTemperature((300-(0.4*(i+1)))*kelvin)
	simulation.step(500)

#not sure if required? minimize the energy of the system and report final configuration
simulation.minimizeEnergy()
pdb_reporter.report(simulation, simulation.context.getState(getPositions=True))

#print out the restraint energy to assess the quality of the structure
state = simulation.context.getState(getEnergy=True, groups={31})
restraint_energy = state.getPotentialEnergy()
print "Final restraint energy:", restraint_energy