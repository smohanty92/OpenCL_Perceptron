##Start time
start_time <- Sys.time()

#Read in input data
D = read.csv('iris.csv')

#Get length of dataframe
DLength = ncol(D)

#slice dataframe into all input data vectors
x = D[,1:DLength-1]

#and label vectors
y = D[,DLength]

#learning rate
n = 0.3

#initialize normal vector, offset, and radius 
w = integer(ncol(x))
b = 0
r = 0

#plot Data before running algorithm
#plot(x)

#Calculate the radius by obtaining the max length of all input data vectors
for (row in 1:nrow(x)) {
	
	sum = 0
	for (col in 1:ncol(x)) {
		
		sum = sum + (x[row,col] * x[row,col])
	
	}
	
	length = sqrt(length)
	
	if(length > r) {
		r = length
	}
	
}

classified = 0
while (classified!=nrow(x)) {
	
	for (row in 1:nrow(x)) {
		
		sum = 0
		
		for (col in 1:ncol(x)) {			
			sum = sum + w[col] * x[row,col]
		}
		
		if (sign(sum-b) != y[row]) {
			
			#Update free parameters
			w = w + n*y[row]*x[row,]
			b = b - n*y[row]*r*r
			
			#Reset classified counter because we have an error
			classified = 0
		} 
		
		#Update classified counter
		classified = classified + 1
		
	}
	
	#Keep plotting linear decision surface
    #abline(b,w)
	
}

#end time
end_time <- Sys.time()

#execution time
end_time - start_time

