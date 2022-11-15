from nn_sf import *

if __name__ == "__main__":

    #{GPU=1} python3 eval_sf_nn.py ./models/model00.pth,./models/mmm00.dat ./input.dat

    # set env variable to run on GPU
    enable_gpu = os.getenv('GPU') != None

    # grab model parameter file
    if len(sys.argv) < 2:
        print("NOTE: no model parameter file provided.")
        exit()

    filenames = [fn for fn in sys.argv[1].strip().split(',') if fn != '']
    print(filenames)
    for filename in filenames:
        if (os.path.isfile(filename) == False):
            print("ERROR: invalid model parameter data files provided.")
            print(" read in: ", filenames)
            exit()

    model_file = filenames[0]
    mmm_file = filenames[1]

    if len(sys.argv) < 3:
        print("NOTE: no sim parameter file provided.")
        exit()

    input_file = sys.argv[2].strip()

    sf_model = SFModel(enable_gpu=enable_gpu,train=False)

    if enable_gpu:
        sf_model.cuda()

    sf_model.load_model(model_file,on_gpu=enable_gpu)
    sf_model.load_mmm(mmm_file)

    # greedy read first two non comment lines
    i = 0
    with open(input_file,'r') as f:
        for line in f: 
            if line.startswith('#'):
                continue
            if i == 0:
                lparams = line.strip()
                i += 1
            elif i == 1:
                lcoeffs = line.strip()
                break
        f.close()

    # now fit with model
    inputs = [float(var) for var in lparams.split(' ')]
    prediction = sf_model.prediction(inputs)
    coeffs = [float(var) for var in lcoeffs.split(' ')]
    print("Prediction: {},{},{},{},{},{},{}".format(*prediction))
    print("Prediction: {},{},{},{},{},{},{}".format(*coeffs))

    # theta distribution for plotting
    thetas = np.linspace(0,1,100) * np.pi/2.0

    # radial coord for prediction (ps) and expected (es)
    ps = np.zeros_like(thetas)
    es = np.zeros_like(thetas)
    for i in range(len(prediction)):
        ps += prediction[i] * thetas**i
    for i in range(len(coeffs)):
        es += coeffs[i] * thetas**i

    tcm = cm.turbo(np.linspace(0.2,0.6,5))

    # and plotting them
    plt.figure(2,figsize=(8,8))
    plt.plot(thetas,ps,'o-',label='prediction',c=tcm[0],mfc='none')
    plt.plot(thetas,es,'.-',label='simulation',c=tcm[2])

    # also map from polar to cartesian to visualise proper shock shape
    # x = r cos(theta)
    # y = r sin(theta)
    xps = -ps * np.cos(thetas)
    yps =  ps * np.sin(thetas)
    xes = -es * np.cos(thetas)
    yes =  es * np.sin(thetas)

    # and plotting them
    plt.figure(3,figsize=(8,8))
    plt.plot(xps,yps,'o-',label='prediction',c=tcm[0],mfc='none')
    plt.plot(xes,yes,'.-',label='simulation',c=tcm[2])

    # it is a bit awkard but we can create a Bezier curve of the surface
    #  from the input params in cartesian space, and then map them to polar


    p = np.zeros((4,2))
    # input params: R2,K1,K2,M,T
    R1 = 1.0
    R2,K1,K2,M,T = inputs
    p[0,:] = [-R1, 0]
    p[1,:] = [-R1, K1*R1]
    p[2,:] = [-K2*R2, R2]
    p[3,:] = [ 0, R2]
    ts = np.linspace(0,1,thetas.shape[0])
    xy = genBezierPoints(p,ts.shape[0])

    # plot in cartesian space
    plt.figure(3)
    plt.plot(xy[:,0],xy[:,1],'k.-',label='surface')
    plt.plot([xy[0,0],xes[0]],[xy[0,1],yes[0]],'-',c=tcm[2],zorder=0)
    plt.plot([xy[-1,0],xes[-1]],[xy[-1,1],yes[-1]],'-',c=tcm[2],zorder=0)

    # and map to polar
    #   theta = arctan(y/x)
    #   r = (x**2 + y**2)**0.5
    stheta = -np.arctan(xy[:,1] / (xy[:,0] - 1e-16))
    sr = (xy[:,0]**2 + xy[:,1]**2)**0.5

    # and plot
    plt.figure(2)
    plt.plot(stheta,sr,'k.-',label='surface')


    abs_error = np.sum(np.abs(ps - es))
    print("Absolute error: {}".format(abs_error))
    l2_error = (np.sum((ps - es)**2) / ps.shape[0])**0.5
    print("L2 error: {}".format(l2_error))

    # visualise the model
    # sf_model.visualise_model(1.5, 5.0, 5.0, 45.0, 100)

    plt.figure(2)
    plt.title('polar coordinates')
    plt.xlim(0,np.pi/2)
    plt.xlabel('theta')
    plt.ylabel('radius')
    plt.xticks([i*np.pi/8 for i in range(5)],
    ['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
    plt.legend(loc='best')

    plt.figure(3)
    plt.title('cartesian coordinates')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')


    if False:
        # HACK
        # grabbed pos data for outer block from a sim to compare:

        # all vertex data for outer block stored like:
        # -1.444847941398620605e+00 0.000000000000000000e+00 0.000000000000000000e+00
        pos = np.loadtxt('tmp.dat')

        # input dimensions:
        # for this block, vertices: (41,6)
        p = np.reshape(pos,(41,6,3))

        plt.figure(3)
        plt.plot(p[:,0,0],p[:,0,1],'s-',c=tcm[2])

# plt.figure(3)
# plt.savefig('examples/ex3/pred_cartesian.png',dpi=300)
# plt.figure(2)
# plt.savefig('examples/ex3/pred_polar.png',dpi=300)

# plt.show(block=False)
plt.show()
