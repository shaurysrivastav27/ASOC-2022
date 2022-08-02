def to_categorical(numclasses,y):
	data = torch.zeros(len(y),numclasses)
	for i in range(len(y)):
		data[i][y[i].long()] = 1
	return data

