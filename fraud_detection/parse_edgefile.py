import pandas as pd
import glob
import os

def get_files(filename_pattern, root_dir):
    return glob.iglob(os.path.join(root_dir, '') + filename_pattern, recursive=True)

def parse_edgelist(edges, dataframe: pd.DataFrame, fn_mapping, source_type='user', sink_type='user'):
    df_extend = pd.read_csv(edges)

    #create map_fn if dtype = object
    if df_extend.dtypes[1] == 'object':
        fn_mapping[df_extend.columns[1]] = {}
        index = 0
        for i in df_extend[df_extend.columns[1]].drop_duplicates():
            fn_mapping[df_extend.columns[1]][i] = index
            index +=1

    src = df_extend.columns[0]
    dst = df_extend.columns[1]
    df_extend['src_type'] = src
    df_extend['dst_type'] = dst
    df_extend['edge_type'] = src + "<>" + dst
    df_extend = df_extend.rename(columns={src: 'src', dst: 'dst'})
    
    #add df_extend to origin dataframe
    dataframe  = pd.concat([dataframe,df_extend], ignore_index= True)
               
    
    return dataframe, fn_mapping

def construct_df(edges):
    df = pd.DataFrame(columns=['src', 'src_type', 'dst', 'dst_type', 'edge_type'])
    # create a mapping function here
    fn_mapping = {}

    # adjust the file DeviceInfo
    for edge in edges.values:
        df, fn_mapping = parse_edgelist('../processed_data/' + edge[0], df, fn_mapping) 

    # generate dst_id
    shape = df[['dst', 'dst_type']].drop_duplicates().shape
    dst_id = df[['dst', 'dst_type']].drop_duplicates()
    dst_id['dst_id'] = range(1, 1 + shape[0])
    df = pd.merge(df, dst_id, on = ['dst','dst_type'])
    return df, fn_mapping

def reconstruct_device_info_file():
    specific_device = ['MacOS', 'Windows', 'iOS Device']
    df = pd.read_csv('./processed_data/relation_DeviceInfo_edgelist.csv')
    for id, device in df.iterrows():
        if device['DeviceInfo'] not in specific_device:
            df['DeviceInfo'].replace(device['DeviceInfo'], 'others', inplace=True)
    df.to_csv('./processed_data/relation_DeviceInfo_edgelist.csv', index= False)