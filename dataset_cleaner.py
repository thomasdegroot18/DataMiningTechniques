import pandas as pd


# Loading in the dataset with pandas
df = pd.read_csv('dataset_mood_smartphone.csv')

# Dropping the id column
df = df.drop(df.columns[[0]], axis=1)

# Remove all NaN
df = df.dropna(0,'any',None,None,False)

# Transform the time column to useful datatime format.
df['time'] = pd.to_datetime(df['time'])

variables = list(df.variable.unique())
ids = df.id.unique()

time_start = df.time.min().normalize()
time_end = df.time.max().normalize()

time_range = pd.date_range(time_start, time_end)

multiindex = pd.MultiIndex.from_product([ids,time_range], names = ["Ids", "Day"])
conversion = pd.DataFrame(0,index = multiindex ,columns= variables)


index_iterator = {}

for time in time_range:
	for id_elem in ids:
		for elem in ["mood", "circumplex.arousal", "circumplex.valence"]:
			index_iterator[(id_elem, time.date(), elem)] = 0
			

for index, row in df.iterrows():
	if index % 10000 == 0:
		print(index)
	if row["variable"] in ["mood", "circumplex.arousal", "circumplex.valence"]:
		conversion.loc[row["id"]].loc[row["time"].date()][row["variable"]] += row["value"]

		index_iterator[(row["id"],row["time"].date(),row["variable"])] += 1
	else:

		conversion.loc[row["id"]].loc[row["time"].date()][row["variable"]] += row["value"] 



for index, row in conversion.iterrows():
	mood_avg = max(1,index_iterator[index[0], index[1].date(),"mood"])
	arousal_avg = max(1,index_iterator[index[0], index[1].date(),"circumplex.arousal"])
	valence_avg = max(1,index_iterator[index[0], index[1].date(),"circumplex.valence"])
	conversion.loc[index,"mood"] = row["mood"] / mood_avg
	conversion.loc[index,"circumplex.arousal"] = row["circumplex.arousal"]/ arousal_avg
	conversion.loc[index,"circumplex.valence"] = row["circumplex.valence"]/ valence_avg

conversion.to_csv('dataset_mood_smartphone_converter.csv')