# standard imports
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import matplotlib.pyplot as plt

from datetime import date
from datetime import datetime
from datetime import timedelta

v2i = {  # variables to index
    # without 'activity' because it is weird, so ignoring it
    "mood": 0,
    "circumplex.arousal": 1,
    "circumplex.valence": 2,
    "screen": 3,
    "call": 4,
    "sms": 5,
    "appCat.builtin": 6,
    "appCat.communication": 7,
    "appCat.entertainment": 8,
    "appCat.finance": 9,
    "appCat.game": 10,
    "appCat.office": 11,
    "appCat.other": 12,
    "appCat.social": 13,
    "appCat.travel": 14,
    "appCat.unknown": 15,
    "appCat.utilities": 16,
    "appCat.weather": 17,
}


class Lstm(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Lstm, self).__init__()
        self.lstm1 = nn.LSTM(n_input, n_hidden)
        self.lstm2 = nn.LSTM(n_hidden, 1)

    def forward(self, sequence, given_hidden_state=None):
        out = []
        if given_hidden_state == None:
            hidden_state_1, hidden_state_2 = None, None
        else:
            hidden_state_1, hidden_state_2 = given_hidden_state
        for sequence_element in sequence.chunk(sequence.size(0), dim=0):
            cell_state_1, hidden_state_1 = self.lstm1(sequence_element, hidden_state_1)
            prediction, hidden_state_2 = self.lstm2(cell_state_1, hidden_state_2)
            out.append(prediction)
        return torch.stack(out).squeeze(1), (hidden_state_1, hidden_state_2)


def train_lstm_tutorial():
    lstm = Lstm(1, 64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)

    for i in range(1000):
        data = np.sin(np.linspace(0, 10, 100) + 2 * np.pi * np.random.rand())
        xs = data[:-1]
        ys = data[1:]
        X = Variable(torch.Tensor(xs).view(-1, 1, 1))
        y = Variable(torch.Tensor(ys))
        optimizer.zero_grad()
        lstm_out, _ = lstm(X)
        loss = criterion(lstm_out[20:].view(-1), y[20:])
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("i {}, loss {}".format(i, loss.data.numpy()[0]))


def make_data():
    with open("sorted_on_person_and_date.csv") as the_file:
        csv_file = csv.reader(the_file)

        last_mood = None
        last_arousal = None
        last_valence = None

        data_series = []  # list of list of lists: one for every person
        person_series = []  # list of lists: one for every day
        data_series_real_moods = []  # list of list of booleans
        person_series_real_moods = []  # list of booleans: indicating whether a mood value was interpolated or real

        def EMPTY_DAILY_DATA():
            return [[], [], []] + [0] * 15

        daily_data = EMPTY_DAILY_DATA()
        current_person = "AS14.01"
        current_day = datetime.strptime("17/02/2014 12:04", "%d/%m/%Y %H:%M").date()

        for row in csv_file:
            _, person, time, variable, value = row

            if variable == 'activity': continue

            day = datetime.strptime(time, "%d/%m/%Y %H:%M").date()

            if day != current_day or person != current_person:
                real_mood = True

                if daily_data[v2i["mood"]] == [] and last_mood is None:
                    daily_data = EMPTY_DAILY_DATA()
                    current_day = day
                else:
                    if daily_data[v2i["mood"]] == [] and last_mood is not None:
                        daily_data[v2i["mood"]] = last_mood
                        daily_data[v2i["circumplex.arousal"]] = last_arousal
                        daily_data[v2i["circumplex.valence"]] = last_valence
                        real_mood = False

                    for i, item in enumerate(daily_data):
                        if type(item) is list:
                            daily_data[i] = sum(item) / len(item)

                    last_mood = daily_data[v2i["mood"]]
                    last_arousal = daily_data[v2i["circumplex.arousal"]]
                    last_valence = daily_data[v2i["circumplex.valence"]]

                    person_series.append(daily_data)
                    person_series_real_moods.append(real_mood)

                    daily_data = EMPTY_DAILY_DATA()
                    current_day = current_day + timedelta(days=1)

                    if person != current_person:  # arrived at a new person
                        data_series.append(person_series)
                        person_series = []

                        data_series_real_moods.append(person_series_real_moods)
                        person_series_real_moods = []

                        current_day = day
                        current_person = person

                        last_mood = None
                        last_arousal = None
                        last_valence = None

                    else:
                        while day != current_day:
                            person_series.append([last_mood, last_arousal, last_valence] + [0] * 15)
                            person_series_real_moods.append(False)
                            current_day = current_day + timedelta(days=1)

            value = 0 if value == '' else float(value)
            if variable in ["mood", "circumplex.arousal", "circumplex.valence"]:
                daily_data[v2i[variable]].append(float(value))
            else:
                daily_data[v2i[variable]] += float(value)

    return (data_series, data_series_real_moods)


def full_experiment():
    EPOCHS = 20

    data_set, real_moods = make_data()

    error_per_person = []

    for left_out_person_id in range(len(data_set)):
        lstm = Lstm(18, 64)
        loss_function = nn.MSELoss()
        optimize_function = optim.Adam(lstm.parameters(), lr=0.001)

        error_per_epoch = []
        print("Testing with leaving out {}".format(left_out_person_id))

        for epoch in range(EPOCHS):
            if epoch % 10 == 0: print("   Epoch {}".format(epoch))

            for i_person, person_series in enumerate(data_set):
                if i_person == left_out_person_id: continue
                inputs = person_series[:-1]
                expected_outputs = []

                for i in range(1, len(person_series)):
                    expected_outputs.append(((person_series[i][v2i['mood']] - 1) / 4.5) - 1)

                torch_inputs = Variable(torch.Tensor(inputs).view(-1, 1, 18))
                torch_expected_outputs = Variable(torch.Tensor(expected_outputs))

                optimize_function.zero_grad()
                lstm_out, _ = lstm(torch_inputs)
                loss = loss_function(lstm_out.view(-1), torch_expected_outputs)
                loss.backward()
                optimize_function.step()

            # Now test on left out dataset

            test_person_series = data_set[left_out_person_id]
            test_real_moods = real_moods[left_out_person_id][1:]  # skip first one
            torch_inputs = Variable(torch.Tensor(test_person_series[:-1]).view(-1, 1, 18))
            expected_outputs = []

            for i in range(1, len(test_person_series)):
                expected_outputs.append(test_person_series[i][v2i['mood']])

            lstm_out, _ = lstm(torch_inputs)
            plain_out = list(lstm_out.data.numpy().flatten())
            scaled_out = [(x + 1) * 4.5 + 1 for x in plain_out]

            test_errors = []

            for i in range(len(test_real_moods)):
                if test_real_moods[i] == True:
                    MEAN_SQUARED_ERROR = True
                    if (MEAN_SQUARED_ERROR):
                        test_errors.append((scaled_out[i] - expected_outputs[i]) ** 2)
                    else:
                        test_errors.append(abs(scaled_out[i] - expected_outputs[i]))

            avg = sum(test_errors) / len(test_errors)
            error_per_epoch.append(avg)

            if epoch % 1 == 0: print("   Avg error: {}".format(avg))

        error_per_person.append(np.array(error_per_epoch))

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


full_experiment()
