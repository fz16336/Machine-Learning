class functions:
    
    def add_gaussian_noise(im,prop,varSigma):
        N = int(np.round(np.prod(im.shape)*prop))
        index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
        e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
        im2 = np.copy(im).astype('float')
        im2[index] += e[index]
        return im2

    def add_saltnpeppar_noise(im,prop):
        N = int(np.round(np.prod(im.shape)*prop))
        index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
        im2 = np.copy(im)
        im2[index] = 1-im2[index]
        return im2

    def y(x):
        y = 2*x
        return y
