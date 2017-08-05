from caffe2.python import core, cnn, net_drawer, workspace, visualize
import numpy as np
from IPython import display
from matplotlib import pyplot

def save_img(img, filename):
    f = open(filename, 'wb')
    f.write(img)
    f.close()

def show_net(net):
    print '-' * 80
    print("Current network proto:\n\n{}".format(net.Proto()))

def show_blob(name):
    blob = workspace.FetchBlob(name)
    print("{} is {}: {}".format(name, type(blob), blob))

def show_all_blobs():
    print '*' * 80
    show_blob('X')
    show_blob('Y_noise')
    show_blob('Y_pred')
    show_blob('dist')
    show_blob('loss')
    show_blob('loss_autogen_grad')
    show_blob('dist_grad')
    show_blob('Y_noise_grad')
    show_blob('Y_pred_grad')
    show_blob('X_grad')
    show_blob('W_grad')
    show_blob('B_grad')
    show_blob('W')
    show_blob('B')
    show_blob('W_gt')
    show_blob('B_gt')

    Y_pred_grad = workspace.FetchBlob('Y_pred_grad')
    if isinstance(Y_pred_grad, str):
        return
    s = 0.0
    for f in Y_pred_grad:
        s = s + float(f)
    print 'dB: %f' % s

    X = workspace.FetchBlob('X')

    dW1 = 0.0
    dW2 = 0.0
    for i in range(len(X)):
        p = X[i]
        y = Y_pred_grad[i][0]
        x1 = p[0]
        x2 = p[1]
        dW1 = dW1 + x1 * y
        dW2 = dW2 + x2 * y
    print('dW: {}'.format([dW1, dW2]))


def test(option, iters):
    init_net = core.Net("init")
    # The ground truth parameters.
    W_gt = init_net.GivenTensorFill(
        [], "W_gt", shape=[1, 2], values=[2.0, 1.5])
    B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
    # Constant value ONE is used in weighted sum when updating parameters.
    ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
    # ITER is the iterator count.
    ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)
    
    # For the parameters to be learned: we randomly initialize weight
    # from [-1, 1] and init bias with 0.0.
    W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
    B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
    print('Created init net.')

    train_net = core.Net("train")
    # First, we generate random samples of X and create the ground truth.
    X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0, run_once=0)
    Y_gt = X.FC([W_gt, B_gt], "Y_gt")
    # We add Gaussian noise to the ground truth
    noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0, std=1.0, run_once=0)
    Y_noise = Y_gt.Add(noise, "Y_noise")
    # Note that we do not need to propagate the gradients back through Y_noise,
    # so we mark StopGradient to notify the auto differentiating algorithm
    # to ignore this path.
    Y_noise = Y_noise.StopGradient([], "Y_noise")
    
    # Now, for the normal linear regression prediction, this is all we need.
    Y_pred = X.FC([W, B], "Y_pred")
    
    # The loss function is computed by a squared L2 distance, and then averaged
    # over all items in the minibatch.
    dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
    loss = dist.AveragedLoss([], ["loss"])

    graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
    img = graph.create_png()
    save_img(img, 'forward.png')
    show_net(train_net)

    # Get gradients for all the computations above.
    gradient_map = train_net.AddGradientOperators([loss])
    graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
    img = graph.create_png()
    save_img(img, 'grad.png')
    show_net(train_net)

    # Increment the iteration by one.
    train_net.Iter(ITER, ITER)
    # Compute the learning rate that corresponds to the iteration.
    LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1,
                                policy="step", stepsize=20, gamma=0.9)
    
    # Weighted sum
    train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
    train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
    
    # Let's show the graph again.
    graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
    img = graph.create_png()
    save_img(img, 'backprop.png')

    show_net(train_net)

    workspace.RunNetOnce(init_net)
    workspace.CreateNet(train_net)

    if option == 1:
        show_all_blobs()
        print('run once')
        workspace.RunNet(train_net.Proto().name)
        show_all_blobs()

        print('run iters')
        for i in range(iters):
            workspace.RunNet(train_net.Proto().name)

        show_all_blobs()
        
    if option == 2:
        workspace.RunNetOnce(init_net)
        w_history = []
        b_history = []
        for i in range(iters):
            workspace.RunNet(train_net.Proto().name)
            w_history.append(workspace.FetchBlob("W"))
            b_history.append(workspace.FetchBlob("B"))

        show_all_blobs()
        w_history = np.vstack(w_history)
        b_history = np.vstack(b_history)
        pyplot.plot(w_history[:, 0], w_history[:, 1], 'r')
        pyplot.axis('equal')
        pyplot.xlabel('w_0')
        pyplot.ylabel('w_1')
        pyplot.grid(True)
        pyplot.figure()
        pyplot.plot(b_history)
        pyplot.xlabel('iter')
        pyplot.ylabel('b')
        pyplot.grid(True)
        pyplot.show()

if __name__ == '__main__':
    import sys
    option = 1
    iters = 50
    if len(sys.argv) > 1:
        option = int(sys.argv[1])
    if len(sys.argv) > 2:
        iters = int(sys.argv[2])
    test(option, iters)
