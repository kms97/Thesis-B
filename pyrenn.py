
import numpy as np
import timeit
from tensorflow.examples.tutorials.mnist import input_data


def CreateNN(nn,dIn=[0],dIntern=[],dOut=[]):
	"""Create Neural Network
	
	Args:
		nn: structure of the NN [I HL1 HL2 ... HLN OL]
			number of layers is the length of the list-1
			number neurons in each layer is the given number
			Example: [2, 3, 4, 1] represents a NN with
				two inputs
				two hidden layers with 3 and 4, respectively
				one linear output layer
		dIn: Time delays for NN inputs. 
			To use only the input of timestep t dIn = [0]
			dIn = [0,1,5] will use inputs of timestep t,t-1 and t-5
		dIntern: Time delays for recurrent internal connections of NN.
			dIntern has to be greater than zero (layer output at timestep t-x)!
			if a non empty list is given, recurrent connection from every
			layer to itself and every layer before is added.
			dIntern = [1,2,5] adds a recurrent connection for the output of
			each layer in timestep t-1,t-2 and t-5
		dOut: Time delays for recurrent connections of output to first hidden layer.
			dOut has to be greater than zero (output at timestep t-x)!
			if a non empty list is given, recurrent connection from NN output
			to first hidden layer is added.
			dOut = [1,3,4] adds a recurrent connection for the output of
			in timestep t-1,t-3 and t-5
	Returns:
		net: untrained neural network
	"""
	
	net = {} #neural network
	delay = {'In':dIn,'Intern':dIntern,'Out':dOut} #Time delays
	net['delay'] = delay #Time delays	
	net['nn'] = nn #structure
	net['M'] = len(nn)-1 #number of layers of the NN
	net['layers'] = nn[1:] #structure without inputs
	net['dmax'] = max(max(dIn,dIntern,dOut)) # maximum time delay	
	net = w_Create(net) #initialize random weight vector and specify sets
	net['w'] = net['w0'].copy() #weight vector used for calculation
	net['N'] = len(net['w0']) #number of weights
	return net
	
def w_Create(net):
	"""	Creates random weight vector of NN and defines sets needed for
		derivative calculation
	
	Args:
		net: neural network
	Returns:
		net: neural network
	"""
	
	M = net['M'] #number of layers
	layers = net['layers'] #NN structure
	delay = net['delay'] #time delays in Network Connections
	inputs = net['nn'][0] #number of inputs
	
	X = []	#set of input layers (input layers or layers with internal delay>0 )
	U = [] 	#set of output layers (output of layer is used for cost function calculation
			# or is added to the input layer with delay>1)
	IW = {}	#input-weight matrices
	LW = {} #LW[m,l,d] connection weight matrix layer m -> layer l with delay d
	b = {} #b[m]: bias vector of layer m
	L_b = {}# L_b[m]: set of layers with a backward connection to layer m
	L_f = {}# L_f[m]: set of layers with a forward connection to layer m
	dL = {}	#dL[m,l]: delays for the connection layer m -> layer l
	CX_LW = {} #CX_LW[u]: set of all input layers, u has a connection to
	CU_LW = {} #CU_LW[x]: set of all output layers, x has a connection to
	b = {} #b[m]: bias vector of layer m
	
	
	'''Inputs'''
	I = {}	#set of inputs with a connection to layer 1
	I[1]=[1] 	#Inputs only connect to layer 1
	
	dI={}	#Delays for connection of input to layer 1
	dI[1,1] = delay['In']
	for d in dI[1,1]:
		IW[1,1,d] = np.random.rand(layers[0],inputs)-0.5 	#input-weight matrix
														#random values [-0.5,0.5]
	X.append(1) 	#first layer is input layer
	
	'''Internal Connection Weight Matrices'''
	for m in range(1,M+1):
		L_b[m] = [] #L_b[m]: Set of layers that have a backward connection to layer m
		L_f[m] = [] #L_f[m]: Set of layers that have a forward connection to layer m
			
		#Forward Connections
		if m>1:
			l=m-1
			dL[m,l]=[0] #no delay for forward connections
			LW[m,l,0] = np.random.rand(layers[m-1],layers[l-1])-0.5 #connection weight matrix
			L_b[l].append(m) #layer m has backward connection to layer l
			L_f[m].append(l) #layer l has forward connection to layer m
	
		#Recursive Connections
		for l in range(m,M+1):
			if (m==1)and(l==M):
			#special case delay form output to layer 1
				dL[m,l]=delay['Out'] #delays from output to first layer
			else:
				dL[m,l]=delay['Intern'] #internal delays
				
			for d in dL[m,l]:
			#all delays for connection l->m
				LW[m,l,d] = np.random.rand(layers[m-1],layers[l-1])-0.5 #connection weight matrix
				if (l not in L_f[m]):
					#add to L_f[m] if not yet done
					L_f[m].append(l) #layer l has forward connection to layer m
				if (l>=m) and(d>0):
				#if it is a recurrent connection
					if (m not in X):
					#and m is not yet in X
						X.append(m) #add m to input layers
					if (l not in U):
					#and l is not yet in U
						U.append(l) #add l to output layers
		
		
		b[m] = np.random.rand(layers[m-1])-0.5#create bias vector for layer m
	
	if M not in U:
		U.append(M) # #add M to output layers if not yet done
		
	for u in U:
		CX_LW[u] = []
		for x in X:
			if (u in L_f[x]) and (np.any(np.array(dL[x,u])>0)) and (x not in CX_LW[u]):
				#if u in L_f[x] and connection x->u has delay>0 and x is not yet in CX_LW[u]
				CX_LW[u].append(x) #add u
	for x in range(1,M+1):
		CU_LW[x] = []
		for u in U:
			try:
				if np.any(np.array(dL[x,u])>0):
				# if connection u -> x has delay >0
					CU_LW[x].append(u)
			except KeyError:
				pass
	#Add to NN
	net['U'] = U
	net['X'] = X
	net['dL'] = dL
	net['dI'] = dI
	net['L_b'] = L_b
	net['L_f'] = L_f
	net['I']=I
	net['CX_LW'] = CX_LW
	net['CU_LW'] = CU_LW
	net['w0'] = Wb2w(net,IW,LW,b)
	return net
	
def Wb2w(net,IW,LW,b):
	"""	Converts Input Weight matrices IW, connection weight matrices LW
		and bias vectors b to weight vector w
	
	Args:
		net: 	neural network
		IW		input-weight matrices
		LW 		LW[m,l,d] connection weight matrix layer m -> layer l with delay d		b		b[m]: bias vector of layer m			
	Returns:
		w: 		weight vector
	"""
	
	dL = net['dL'] #dL[m,l]: delays for the connection layer m -> layer l
	dI = net['dI'] #Delays for connection of input to layer 1
	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	
	w = np.array([]) #empty weight vector
	
	for m in range(1,M+1):
		#input weights
		if m==1:
			for i in I[m]:
				for d in dI[m,i]:
					w = np.append(w,IW[m,i,d].flatten('F'))
		#internal connection weights
		for l in L_f[m]:
			for d in dL[m,l]:
				w = np.append(w,LW[m,l,d].flatten('F'))
		#bias weights
		w = np.append(w,b[m])
	
	return w
	
def w2Wb(net):
	"""	Converts weight vector w to Input Weight matrices IW, connection weight matrices LW
		and bias vectors b
	
	Args:
		net: 	neural network (containing weight vector w)		
	Returns:
		IW		input-weight matrices
		LW 		LW[m,l,d] connection weight matrix layer m -> layer l with delay d		b		b[m]: bias vector of layer m	
	"""
	
	dL = net['dL'] #dL[m,l]: delays for the connection layer m -> layer l
	dI = net['dI'] #Delays for connection of input to layer 1
	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	layers = net['layers'] #structure of the NN
	inputs = net['nn'][0] #number of inputs
	w_temp = net['w'].copy() #weight vector
		
	IW = {}	#input-weight matrices
	LW = {} #LW[m,l,d] connection weight matrix layer m -> layer l with delay d
	b = {} #b[m]: bias vector of layer m
	
	for m in range(1,M+1):
	
		#input weights
		if m==1:
			for i in I[m]:
				for d in dI[m,i]:
					w_i = inputs*layers[m-1]
					vec =w_temp[0:w_i]
					w_temp = w_temp[w_i:]
					IW[m,i,d] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		#internal connection weights
		for l in L_f[m]:
			for d in dL[m,l]:
				w_i = layers[l-1]*layers[m-1]
				vec =w_temp[0:w_i]
				w_temp = w_temp[w_i:]
				LW[m,l,d] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		#bias weights
		w_i = layers[m-1]
		b[m] =w_temp[0:w_i]
		w_temp = w_temp[w_i:]

	return IW,LW,b


def NNOut_(P,net,IW,LW,b,a={},q0=0):
	"""	Calculates NN Output for given Inputs P
		For internal use only
	
	Args:
		P:		NN Inputs
		net: 	neural network
		IW:		input-weight matrices
		LW:		LW[m,l,d] connection weight matrix layer m -> layer l with delay d		b:		b[m]: bias vector of layer m
		a:		Layer Outputs of NN. for use of known historical data
		q0:		Use data starting from datapoint q0 P[q0:]
	Returns:
		Y_NN: 	Neural Network output for input P
		a:		Layer Outputs of NN
		n:		sum output of layers
	"""
	
	dL = net['dL'] #dL[m,l]: delays for the connection layer m -> layer l
	dI = net['dI'] #Delays for connection of input to layer 1
	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	outputs = net['nn'][-1] #number of outputs
	
	n = {} #sum output of layers
	Q = P.shape[1] #number of input datapoints

		
	Y_NN = np.zeros((outputs,Q)) #NN Output
	
	for q in range(q0+1,Q+1):
	#for all datapoints
		a[q,1]=0
		for m in range(1,M+1):
		#for all layers m
			n[q,m]=0 #sum output datapoint q, layer m
			
			#input weights
			if m==1:
				for i in I[m]:
					for d in dI[m,i]:
						if (q-d)>0:
							n[q,m]=np.around(n[q,m]+np.dot(IW[m,i,d],P[:,q-d-1]),decimals=4)
			#connection weights
			for l in L_f[m]:
				for d in dL[m,l]:
					if (q-d)>0:
						n[q,m]=np.around(n[q,m]+np.dot(LW[m,l,d],a[q-d,l]),decimals=4)
			#bias
			n[q,m]=n[q,m] + b[m]
			
			#Calculate layer output
			if m==M:
				a[q,M]=n[q,M] #linear layer for output
			else:
				a[q,m] = np.around(np.tanh(n[q,m]),decimals=4)
		Y_NN[:,q-1]=a[q,M]
	Y_NN = Y_NN[:,q0:]
	return Y_NN,n,a
	
def NNOut(P,net,P0=None,Y0=None):
	"""	Calculates NN Output for given Inputs P
		User Function
	Args:
		P:		NN Inputs
		net: 	neural network
		P0:		previous input Data
		Y0:		previous output Data
	Returns:
		Y_NN: 	Neural Network output for input P
	"""
	Y=np.zeros((net['layers'][-1],int(np.size(P)/net['nn'][0])))
	data,net = prepare_data(P,Y,net,P0=P0,Y0=Y0)
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	Y_NN = NNOut_(data['P'],net,IW,LW,b,a=data['a'],q0=data['q0'])[0]
	
	#scale normalized Output
	Y_NN_scaled = Y_NN.copy()
	for y in range(np.shape(Y_NN)[0]):
		Y_NN_scaled[y] = Y_NN[y]*net['normY'][y]
	
	if np.shape(Y_NN_scaled)[0]==1:
		Y_NN_scaled=Y_NN_scaled[0]
	return Y_NN_scaled
	
	
def RTRL(net,data):
	"""	Implementation of the Real Time Recurrent Learning Algorithm based on:
		Williams, Ronald J.; Zipser, David: A Learning Algorithm for Continually Running
		Fully Recurrent Neural Networks. In: Neural Computation, Nummer 2, Vol. 1
		(1989), S. 270-280.
		
	Args:
		net:	neural network
		data: 	Training Data
	Returns:
		J: 		Jacobian Matrix. derivatives of e with respect to the weight vector w
		E:		Mean squared Error of the Neural Network compared to Training data
		e:		error vector: difference of NN Output and target data['Y']
	"""
	P = data['P']	#Training data Inputs
	Y = data['Y']	#Training data Outputs
	a = data['a']	#Layer Outputs
	q0 = data['q0']	#Use training data [q0:]
	
	
	dL = net['dL'] #dL[m,l]: delays for the connection layer m -> layer l
	dI = net['dI'] #Delays for connection of input to layer 1
	I = net['I'] #set of inputs with a connection to layer 1
	L_f = net['L_f'] # L_f[m]: set of layers with a forward connection to layer m
	L_b = net['L_b'] # L_f[m]: set of layers with a forward connection to layer m
	M = net['M'] #number of layers of NN
	inputs = net['nn'][0] #number of inputs
	outputs = net['nn'][-1] #number of outputs
	layers = net['layers'] #structure of the NN
	max_delay  = net['dmax'] # Maximum delay in the NN
	U = net['U'] #set of input layers (input layers or layers with internal delay>0 )
	X = net['X'] #set of output layers (output of layer is used for cost function calculation
			# or is added to the input layer with delay>1)
	CU_LW = net['CU_LW'] #CU_LW[x]: set of all output layers, x has a connection to		
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	
	########################
	# 1. Calculate NN Output
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. Calculate Cost function E
	Y_delta = Y - Y_NN #error matrix
	e = np.around(np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0], decimals=4) #error vector
	E = np.around(np.dot(e,e.transpose()), decimals=4) #Cost function (mean squared error)
	
	#########################
	# 3. Backpropagation RTRL
	
	Q = P.shape[1] #number of input datapoints
	Q0 = Q-q0 #number of datapoints without "old data"
	
	#Definitions
	dAu_db = {}		#derivative of layer output a(u) with respect to bias vector b
	dAu_dIW = {}	#derivative of layer output a(u) with respect to input weights IW
	dAu_dLW = {}	#derivative of layer output a(u) with respect to connections weights LW
	dA_dw = {}		#derivative of layer outputs a with respect to weight vector w
	S = {}			#Sensitivity Matrix
	Cs = {}			#Cs[u]: Set of layers m with an existing sensitivity matrix S[q,u,m]
	CsX = {}		#CsX[u]: Set of input layers x with an existing sensitivity matrix
					#S[q,u,x]
					#Cs and CsX are generated during the Backpropagation
			
	#Initialize
	J = np.zeros((Q0*layers[-1],net['N']))	#Jacobian matrix
	for q in range(1,q0+1):
		for u in U:
			dAu_dLW[q,u] = np.zeros((layers[u-1],net['N']))
	
	###
	#Begin RTRL
	for q in range(q0+1,Q+1):
	
		#Initialize
		U_ = [] #set needed for calculating sensitivities
		for u in U:
			Cs[u] = []
			CsX[u] = []
			dA_dw[q,u] = 0
			
		#Calculate Sensitivity Matrices
		for m in range(M,1-1,-1):
		# decrement m in backpropagation order
		
			for u in U_:
				S[q,u,m] = 0 #Sensitivity Matrix layer u->m
				for l in L_b[m]:
					S[q,u,m] = np.around(S[q,u,m] \
						+ np.dot(np.dot(S[q,u,l],LW[l,m,0]),np.diag(1-(np.tanh(n[q,m]))**2)),decimals=4)
						#recursive calculation of Sensitivity Matrix layer u->m
				if m not in Cs[u]:
					Cs[u].append(m) #add m to set Cs[u]
					if m in X:
						CsX[u].append(m) #if m ind X, add to CsX[u]
			
			if m in U:
				if m==M:
					#output layer is linear, no transfer function
					S[q,m,m] = np.around(np.diag(np.ones(outputs)),decimals=4) #Sensitivity Matrix S[M,M]
				else:
					S[q,m,m] = np.around(np.diag(1-(np.tanh(n[q,m]))**2),decimals=4) #Sensitivity Matrix S[m,m]
				
				U_.append(m) #add m to U'
				Cs[m].append(m) #add m to Cs
				if m in X:
					CsX[m].append(m) #if m ind X, add to CsX[m]
		
		#Calculate derivatives
		for u in sorted(U):
			#static derivative calculation
			dAe_dw = np.empty((layers[u-1],0)) #static derivative vector: explicit derivative layer outputs with respect to weight vector
			for m in range(1,M+1):
				#Input weights
				if m==1:
					for i in I[m]:
						for d in dI[m,i]:
							if ((q,u,m) not in S.keys()) or (d>=q):
							#if no sensivity matrix exists or d>=q: derivative is zero
								dAu_dIW[m,i,d] = \
									np.around(np.kron(P[:,q-d-1].transpose(),\
											np.zeros((layers[u-1],layers[m-1]))), decimals=4)
							else:
								#derivative output layer u with respect to IW[m,i,d]
								dAu_dIW[m,i,d] = \
									np.around(np.kron(P[:,q-d-1].transpose(),S[q,u,m]),decimals=4)
							dAe_dw = np.append(dAe_dw,dAu_dIW[m,i,d],1) #append to static derivative vector
	
				#Connection weights
				for l in L_f[m]:
					for d in dL[m,l]:
						if ((q,u,m) not in S.keys()) or (d>=q):
						#if no sensivity matrix exists or d>=q: derivative is zero
							dAu_dLW[m,l,d] = \
								np.around(np.kron(a[q,l].transpose(),\
										np.zeros((layers[u-1],layers[m-1]))), decimals=4)
						else:
							dAu_dLW[m,l,d] = \
								np.around(np.kron(a[q-d,l].transpose(),S[q,u,m]),decimals=4)
								#derivative output layer u with respect to LW[m,i,d]
						dAe_dw = np.append(dAe_dw,dAu_dLW[m,l,d],1) #append to static derivative vector
				
				#Bias weights
				if ((q,u,m) not in S.keys()):
					dAu_db[m] = np.zeros((layers[u-1],layers[m-1])) #derivative is zero
				else:
					dAu_db[m] = S[q,u,m] #derivative output layer u with respect to b[m]
				dAe_dw = np.append(dAe_dw,dAu_db[m],1) #append to static derivative vector
				
			#dynamic derivative calculation
			dAd_dw=0 #dynamic derivative, sum of all x
			for x in CsX[u]:
				sum_u_ = 0 #sum of all u_
				for u_ in CU_LW[x]:
					sum_d = 0 #sum of all d
					for d in dL[x,u_]:
						if (q-d>0)and(d>0):
						#delays >0 and <q
							sum_d = np.around(sum_d + np.dot(LW[x,u_,d],dA_dw[q-d,u_]), decimals=4)
					sum_u_ = sum_u_+sum_d
				if sum_u_ is not 0:
					dAd_dw = np.around(dAd_dw + np.dot(S[q,u,x],sum_u_), decimals=4) #sum up dynamic derivative
					
			#static + dynamic derivative
			dA_dw[q,u] = dAe_dw + dAd_dw # total derivative output layer u with respect to w
			
		# Jacobian Matrix
		J[range(((q-q0)-1)*outputs,(q-q0)*outputs),:] = -dA_dw[q,M]
		
		# Delete entries older than q-max_delay in dA_dw
		if q > max_delay:
			new_dA_dw = dict(dA_dw)
			for key in dA_dw.keys():
				if key[0] == q-max_delay:
					del new_dA_dw[key]
			dA_dw = new_dA_dw
		
		# Reset S
		S = {}
		
	return J,E,e

def train_LM(P,Y,net,k_max=100,E_stop=1e-10,dampfac=3.0,dampconst=10.0,\
			verbose = False):
	"""	Implementation of the Levenberg-Marquardt-Algorithm (LM) based on:
		Levenberg, K.: A Method for the Solution of Certain Problems in Least Squares.
		Quarterly of Applied Mathematics, 2:164-168, 1944.
		and
		Marquardt, D.: An Algorithm for Least-Squares Estimation of Nonlinear Parameters.
		SIAM Journal, 11:431-441, 1963.
		
	Args:
		P:		NN Inputs
		Y:		NN Targets
		net: 	neural network
		k_max:	maxiumum number of iterations
		E_stop:	Termination Error, Training stops when the Error <= E_stop
		dampconst:	constant to adapt damping factor of LM
		dampfac:	damping factor of LM
	Returns:
		net: 	trained Neural Network 
	"""
	#create data dict
	data,net = prepare_data(P,Y,net)
	
	#Calculate Jacobian, Error and error vector for first iteration
	J,E,e = RTRL(net,data)
	k = 0
	ErrorHistory=np.zeros(k_max+1) #Vektor for Error hostory
	ErrorHistory[k]=E
	if verbose:
		print('Iteration: ',k,'		Error: ',E,'	scale factor: ',int(dampfac))
	
	while True:
	#run loop until either k_max or E_stop is reached

		JJ = np.rint(np.dot(J.transpose(),J)).astype(np.int8) #J.transp * J
		w = net['w'] #weight vector
		while True:
		#repeat until optimizing step is successful
			
			#gradient
			g = np.around(np.dot(J.transpose(),e),decimals=4)
			
			#calculate scaled inverse hessian
			try:
				G = np.around(np.linalg.inv(JJ+dampfac*np.eye(net['N'])),decimals=4) #scaled inverse hessian
			except np.linalg.LinAlgError:
				# Not invertible. Go small step in gradient direction
				w_delta = np.around(1.0/1e10 * g,decimals=4)
			else:
				# calculate weight modification
				w_delta = np.around(np.dot(-G,g),decimals=4)
			
			net['w'] = w + w_delta #new weight vector
			
			Enew = calc_error(net,data) #calculate new Error E
			
			if Enew<E:
			#Optimization Step successful!
				dampfac = dampfac/dampconst #adapt scale factor
				break #go to next iteration
			else:
			#Optimization Step NOT successful!
				dampfac = dampfac*dampconst#adapt scale factor
		
		#Calculate Jacobian, Error and error vector for next iteration
		J,E,e = RTRL(net,data)
		k = k+1
		ErrorHistory[k] = E
		if verbose:
			print('Iteration: ',k,'		Error: ',E,'	scale factor: ',int(dampfac))
	
		#Ceck if termination condition is fulfilled
		if k>=k_max:
			print('Maximum number of iterations reached')
			break
		elif E<=E_stop:
			print('Termination Error reached')
			break
	
	net['ErrorHistory'] = ErrorHistory[:k]
	return net
	
	
def calc_error(net,data):
	"""	Calculate Error for NN based on data
		
	Args:
		net:	neural network
		data: 	Training Data
	Returns:
		E:		Mean squared Error of the Neural Network compared to Training data
	"""
	P = data['P']	#Training data Inputs
	Y = data['Y']	#Training data Outputs
	a = data['a']	#Layer Outputs
	q0 = data['q0']	#Use training data [q0:]
	
	IW,LW,b = w2Wb(net) #input-weight matrices,connection weight matrices, bias vectors
	
	########################
	# 1. Calculate NN Output
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. Calculate Cost function E
	Y_delta = Y - Y_NN #error matrix
	e = np.around(np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0],decimals=4)#error vector	
	E = np.around(np.dot(e,e.transpose()),decimals=4) #Cost function (mean squared error)
	return E
	
def prepare_data(P,Y,net,P0=None,Y0=None):
	"""	Prepare Input Data for the use for NN Training and check for errors
		
	Args:
		P:		neural network Inputs
		Y: 		neural network Targets
		net: 	neural network
		P0:		previous input Data
		Y0:		previous output Data
	Returns:
		data:	dict containing data for training or calculating putput
	"""	
	
	#Convert P and Y to 2D array, if 1D array is given
	if P.ndim==1:
		P = np.array([P])
	if Y.ndim==1:
		Y = np.array([Y])
		
	#Ceck if input and output data match structure of NN	
	if np.shape(P)[0] != net['nn'][0]:
		raise ValueError("Dimension of Input Data does not match number of inputs of the NN")
	if np.shape(Y)[0] != net['nn'][-1]:
		raise ValueError("Dimension of Output Data does not match number of outputs of the NN")
	if np.shape(P)[1] != np.shape(Y)[1]:
		raise ValueError("Input and output data must have same number of datapoints Q")
	
	#check if previous data is given
	if (P0 is not None) and (Y0 is not None):

		#Convert P and Y to 2D array, if 1D array is given
		if P0.ndim==1:
			P0 = np.array([P0])
		if Y0.ndim==1:
			Y0 = np.array([Y0])
			
		#Ceck if input and output data match structure of NN
		if np.shape(P0)[0] != net['nn'][0]:
			raise ValueError("Dimension of previous Input Data P0 does not match number of inputs of the NN")
		if np.shape(Y0)[0] != net['nn'][-1]:
			raise ValueError("Dimension of previous Output Data Y0 does not match number of outputs of the NN")
		if np.shape(P0)[1] != np.shape(Y0)[1]:
			raise ValueError("Previous Input and output data P0 and Y0 must have same number of datapoints Q0")

		q0 = np.shape(P0)[1]#number of previous Datapoints given 
		a = {} #initialise layer outputs
		for i in range(1,q0+1):
			for j in range(1,net['M']):
				a[i,j]=np.zeros(net['nn'][j]) #layer ouputs of hidden layers are unknown -> set to zero
			a[i,net['M']]=Y0[:,i-1]/net['normY'] #set layer ouputs of output layer

		#add previous inputs and outputs to input/output matrices
		P_ = np.concatenate([P0,P],axis=1)
		Y_ = np.concatenate([Y0,Y],axis=1)
	else:
		#keep inputs and outputs as they are and set q0 and a to default values
		P_ = P.copy()
		Y_ = Y.copy()
		q0=0
		a={}
	#normalize
	P_norm = P_.copy()
	Y_norm = Y_.copy()
	if 'normP' not in net.keys():
		normP = np.ones(np.shape(P_)[0])
		for p in range(np.shape(P_)[0]):
			normP[p] = np.max([np.max(np.abs(P_[p])),1.0])
			P_norm[p] = P_[p]/normP[p]
		normY = np.ones(np.shape(Y_)[0])
		for y in range(np.shape(Y_)[0]):
			normY[y] = np.max([np.max(np.abs(Y_[y])),1.0])
			Y_norm[y] = Y_[y]/normY[y]
		net['normP'] = normP
		net['normY'] = normY
	else:
		for p in range(np.shape(P_)[0]):
			P_norm[p] = P_[p]/net['normP'][p]
		normY = np.ones(np.shape(Y)[0])
		for y in range(np.shape(Y_)[0]):
			Y_norm[y] = Y_[y]/net['normY'][y]
		
	#Create data dict
	data = {}		
	data['P'] = P_norm
	data['Y'] = Y_norm
	data['a'] = a
	data['q0'] = q0
	
	return data,net

def saveNN(net,filename):
	"""	Save neural network object to file
		
	Args:
		net: 	neural network object
		filename:	path of csv file to save neural network
	
	"""	
	import csv
	import pandas as pd
	
	#create csv write
	file = open(filename,"w")
	writer = csv.writer(file, lineterminator='\n')

	
	#write network structure nn
	writer.writerow(['nn'])
	writer.writerow(net['nn'])
	
	#write input delays dIn
	writer.writerow(['dIn'])
	writer.writerow(net['delay']['In'])
	
	#write internal delays dIntern
	writer.writerow(['dIntern'])
	if not net['delay']['Intern']:
		writer.writerow(['',''])
	else:
		writer.writerow(net['delay']['Intern'])
		
	#write output delays dIOut
	writer.writerow(['dOut'])
	if not net['delay']['Out']:
		writer.writerow(['',''])
	else:
		writer.writerow(net['delay']['Out'])
		
	#write factor for input data normalization normP
	writer.writerow(['normP'])
	writer.writerow(net['normP'])
	
	#write factor for output data normalization normY
	writer.writerow(['normY'])
	writer.writerow(net['normY'])
	
	#write weight vector w
	writer.writerow(['w'])
	file.close()
	
	file = open(filename,"ab")
	np.savetxt(file,net['w'],delimiter=',',fmt='%.55f')
	
	#close file
	file.close()
	
	return
	
def loadNN(filename):
	"""	Load neural network object from file
		
	Args:
		filename:	path to csv file to save neural network
	Returns:
		net: 	neural network object
	"""	
	import csv
	import pandas as pd
	
	#read csv
	data= list(csv.reader(open(filename,"r")))

	#read network structure nn
	nn = list(np.array(data[1],dtype=np.int))
	
	#read input delays dIn
	dIn = list(np.array(data[3],dtype=np.int))
	
	#read internal delays dIntern
	if data[5] == ['','']:
		dIntern = []
	else:
		dIntern = list(np.array(data[5],dtype=np.int))
		
	#read output delays dIOut
	if data[7] == ['','']:
		dOut = []
	else:
		dOut = list(np.array(data[7],dtype=np.int))
	
	#read factor for input data normalization normP
	normP = np.array(data[9],dtype=np.float)
	
	#read factor for output data normalization normY
	normY = np.array(data[11],dtype=np.float)
	
	#read weight vector w
	w = pd.read_csv(filename,sep=',',skiprows=range(12))['w'].values
	
	#Create neural network and assign loaded weights and factors
	net = CreateNN(nn,dIn,dIntern,dOut)
	net['normP'] = normP
	net['normY'] = normY
	net['w'] = w
	
	return net

if __name__ == "__main__":
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	#train_set, valid_set, test_set = pickle.load( open( "mnist.pkl", "rb" ),encoding='latin1')
	train_set = mnist.train
	train_images = train_set.images.T
	train_labels = (train_set.labels.T)
	test_set = mnist.test
	test_images = test_set.images.T
	test_labels = (test_set.labels.T)

	print ('x_train Examples Loaded = ' + str(train_images.shape))
	print ('y_train Examples Loaded = ' + str(train_labels.shape))
	print ('x_train Examples Loaded = ' + str(test_images.shape))
	print ('y_train Examples Loaded = ' + str(test_labels.shape))

	net = CreateNN([28*28,10,10])
	batch_size = 100
	number_of_batches=100
	for i in range(number_of_batches):
	    r = np.random.randint(0,55000-batch_size)
	    X = train_images[:,r:r+batch_size]
	    Y = train_labels[:,r:r+batch_size]
	    start_time = timeit.default_timer()
	    #Train NN with training data Ptrain=input and Ytrain=target
	    #Set maximum number of iterations k_max
	    #Set termination condition for Error E_stop
	    #The Training will stop after k_max iterations or when the Error <=E_stop
	    net = train_LM(X,Y,net,verbose=True,k_max=3,E_stop=1e-4,dampfac=3)
	    end_time = timeit.default_timer()
	    print("Time Taken: ", (end_time-start_time))
	    r = np.random.randint(0,55000-batch_size)
	    P_ = train_images[:,r:r+batch_size]
	    L_ = train_labels[:,r:r+batch_size]
	    Y_ = NNOut(P_, net)
	    correct = 0
	    for j in range(batch_size):
	    	y_ = np.argmax(Y_[:,j])
	    	l_ = np.argmax(L_[:,j])
	    	if y_ == l_:
	    		correct = correct+1
	    print('Validation Error %: ', (1-correct/batch_size)*100)
	    print('Batch No. ',i,' of ',number_of_batches)


	#test_set = mnist.test
	num = 10000
	#for i in range(num):
	P_ = test_images[:,0:num]
	L_ = test_labels[:,0:num]
	Y_ = NNOut(P_,net)
	correct = 0
	for i in range(num):	
		y_ = np.argmax(Y_[:,i])
		l_ = np.argmax(L_[:,i])
		if y_ == l_:
			correct = correct+1
	print(correct)