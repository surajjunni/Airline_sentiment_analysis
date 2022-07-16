from typing import Optional
from typing import List
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
MODEL_PATH = 'finalized_model.hdf5'
tfidf_pth='tfidf.pickle'
from main import *
clf = pickle.load(open(MODEL_PATH, 'rb'))
vectorizer = pickle.load(open(tfidf_pth, 'rb'))
class Item(BaseModel):
      tweet: str
      class Config:
        schema_extra = {
            "example": {
                "tweet": "@VirginAmerica it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp; they have little recourse"
            }
        }

    

app = FastAPI()

@app.post("/tweets/")
async def create_item(item: Item):
	  data=preprocess(np.array([item.tweet]))
	  data=vectorizer.transform(data).toarray()
	  #print(data)
	  res=clf.predict(data).tolist()
	  res={'result':res[0]}
	  #print(res)
	  return res