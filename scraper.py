import tweepy
import pandas as pd
import time

CONSUMER_KEY = "4ivot00X5xCCaIRzKMQfGpEim"
CONSUMER_SECRET = "YWYNadE3yTy5FBl9zl9MFqmpIKjlp3S9ewXEwbMuZpEIPZDx2b"
ACCESS_TOKEN = "716665023213412353-iaRSPZRAnCaNe7zGji4rfdY5hQsj0RV"
ACCESS_TOKEN_SECRET = "Akk0OWWGzNx2fBIe6smkgFsXI6fsT82kjVQE0o1Y0Voek"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
limit = api.rate_limit_status()

data = pd.read_csv("Twitter/data2.csv", index_col=0)
processed_file = "result2.csv"

data['Text'] = pd.Series(index=data.index)

for i in range(0,len(data)):
	index_no = i + 700
	tweet_id = data['Tweet ID'][index_no]
	
	try:
		status = api.get_status(int(tweet_id))
		data['Text'][index_no] = status.text
	except:
		pass

	print(i)
	time.sleep(5.5)

data.to_csv(processed_file)

#start = 0
#end = 700
#final_file = False
#file_number = 1

# while True:

# 	if end > len(data):
# 		end = len(data) - 1
# 		final_file = True

# 	file_name = "data" + str(file_number) + ".csv"
# 	print("FILE INDEXES: ",start," to ", end)
# 	df_file = data[start:end]
# 	df_file.to_csv(file_name)
# 	print(file_name, " SAVED SUCCESSFULLY")

# 	if final_file:
# 		break
# 	else:
# 		start = end 
# 		end += 700
# 		file_number += 1

#start = time.time()

# for id in tweet_id_list:
# 	try:
# 		status = api.get_status(int(id))
# 		available += 1
# 	except:
# 		removed += 1