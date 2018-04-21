import torch
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import date
from datetime import datetime
from datetime import timedelta


def showPlot(error_per_person):

    for series in error_per_person:
        plt.plot(series, alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.ylabel('average error')
        plt.xlabel('epoch')

    mean = np.mean(np.array(error_per_person), axis=0)
    plt.plot(mean, linewidth=3.0, color='k')

    experiment_name = 'experiment_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    plt.savefig("{}.png".format(experiment_name))

    with open("{}.txt".format(experiment_name), "w") as file:
        output = lambda text: print(text, file=file)

        output("Mean per epoch of all folds:")
        output(mean)
        output("")

        for (i, series) in enumerate(error_per_person):
            output("Average error per epoch when leaving person {} out:".format(i))
            output(series)
            output("")

    plt.show()



# Read in data
def data_input():
    df = pd.read_csv('dataset_mood_smartphone_converter.csv')

    _id = 0
    _moode = 0
    _arousal = 0
    _valence = 0
    check_false_value = False


    df.drop(columns=["activity"])

    # df.loc[:,"mood":] = 2*((df.loc[:,"mood":] - df.loc[:,"mood":].min()) / (df.loc[:,"mood":].max() - df.loc[:,"mood":].min())) - 1

    data = np.zeros((df.shape))

    numpy_id = 0
    day = 0

    for index, row in df.iterrows():
        if _id != row["Ids"]:
            numpy_id += 1
            day = 1
        if _id != row["Ids"] or check_false_value:
            _id = row["Ids"]
            _moode = row["mood"]
            _arousal = row["circumplex.arousal"]
            _valence = row["circumplex.valence"]
            if _moode == 0 and _arousal == 0 and _valence == 0:
                check_false_value = True
                continue
            else:
                check_false_value = False

        if row["mood"] == 0:
            _moode = _moode
            _id = row["Ids"]
            if row["circumplex.arousal"] == 0:
                _arousal = _arousal
            if row["circumplex.valence"] == 0:
                _valence = _valence

        else:
            _id = row["Ids"]
            _moode = row["mood"]
            _arousal = row["circumplex.arousal"]
            _valence = row["circumplex.valence"]


        row["mood"] = _moode
        row["circumplex.arousal"] = _arousal
        row["circumplex.valence"] = _valence
        row["Ids"] = numpy_id
        row["Day"] = day

        day += 1

        data[index,:] = row


    # Normalized and cleansed
    data = data[~np.all(data == 0, axis=1)]

    data_not_temp = data

    i = 1
    row_index = 0
    id_row = 0
    for row in data:
        if row[0] != id_row:
            i = 1
            id_row = row[0]
        data_not_temp[i,2:5] = np.mean(data[row_index:row_index+min(i,5),2:5], axis=0)
        data_not_temp[i,5:] = np.sum(data[row_index:row_index+min(i,5),5:], axis=0)
        row_index += 1
        i+= 1


    data_not_temp[:,3:] = ((data_not_temp[:,3:] - np.amin(data_not_temp[:,3:], axis=0)) / (np.amax(data_not_temp[:,3:], axis=0) - np.amin(data_not_temp[:,3:], axis=0)))

    return data_not_temp




def model_init(batch_size, input_dim, h_dim, h_layers, output_dim, learning_rate):
    model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, h_dim),
    torch.nn.Sigmoid(),
    torch.nn.Linear(h_dim, output_dim),
    )

    loss_fn = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer




def train(input_variable, target_variable, model, loss_fn, optimizer):
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.

    target_variable = (target_variable) / 10

    x = Variable(torch.from_numpy(input_variable).float())
    if target_variable.shape[0] == 1:
        y = Variable(torch.from_numpy([float(target_variable)]), requires_grad=False)
    else:
        y = Variable(torch.from_numpy(target_variable).float(), requires_grad=False)

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data[0]



def test(input_variable, model):
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.

    x = Variable(torch.from_numpy(input_variable).float())

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    return y_pred


def main():
    epochs = 10
    batch_size = 32
    input_dim = 18 
    h_dim = 100
    h_layers = 1 
    output_dim = 1
    learning_rate = 1e-4
    print_every=1000
    plot_every=100

    model, loss_fn, optimizer = model_init(batch_size, input_dim, h_dim, h_layers, output_dim, learning_rate)

    dataset = data_input()

    print_loss_total = 0

    error_per_person = []


    for i in set(dataset[:,0]):
        trainset = dataset[dataset[:,0]!=i,:]
        testset = dataset[dataset[:,0]==i,:]

        error_person = []


        counter = 0

        for epoch in range(0, epochs):

            for iterator in range(1,int(trainset.shape[0]/batch_size)):
                input_variable = trainset[batch_size*(iterator-1):batch_size*iterator,3:]
                target_variable = trainset[batch_size*(iterator-1):batch_size*iterator,2]
                print_loss = train(input_variable, target_variable, model, loss_fn, optimizer)

                # print('the average loss is %.4f' %  (print_loss))

                counter += 32




            print_loss = 0
            test_errors = []

            for iterator in range(1,int(testset.shape[0]/batch_size)+1):
                input_variable = testset[batch_size*(iterator-1):batch_size*iterator,3:]
                target_variable = trainset[batch_size*(iterator-1):batch_size*iterator,2]

                predicted_mood = test(input_variable, model)

                predicted_mood = (predicted_mood) * 10

                if predicted_mood.shape[0] == 1:
                    test_errors.append(abs(float(predicted_mood) - target_variable))
                else:
                    test_errors += list(abs(predicted_mood.data.numpy().flatten() - target_variable))




            average_errors = sum(test_errors) / len(test_errors)
            print(average_errors)
            error_person.append(average_errors)

        error_per_person.append(error_person)


    showPlot(error_per_person)


main()