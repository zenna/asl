import asl
import pandas as pd
import os
import stdargs

def extension(path):
  filename, file_extension = os.path.splitext(path)
  return file_extension


def isopt(path):
  return extension(path) == ".pkl"


"Loads all rundata files in `searchdir`"
def walkoptdata(searchdir):
  return walkload(searchdir, isopt, asl.load_opt)


def walkload(searchdir, isgood, loaddata):
  data = []
  for root, dir, files in os.walk(searchdir, followlinks=True):
    for file in files:
      if isgood(file):
        print(root)
        data.append(loaddata(os.path.join(root, file)))

  return data

# Get all the data frames where the number of items is 1, find the optimal loss

def get_df_where_opt(filter_func, df_to_opt):
  ...
  # {df, opt for df in df_to_opt.items() if 


def optimalloss(searchdir, nitems):
  def nitemsisnitems(opt):
    return opt["nitems"] == nitems
  get_df_where_opt(nitemsisnitems)