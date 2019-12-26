import subprocess
datasetDict = {
      'mesos': 'apache'
    , 'usergrid': 'apache'
    , 'appceleratorstudio': 'appcelerator'
    , 'aptanastudio': 'appcelerator'
    , 'titanium': 'appcelerator'
    , 'duracloud': 'duraspace'
    , 'bamboo': 'jira'
    , 'clover': 'jira'
    , 'jirasoftware': 'jira'
    , 'moodle': 'moodle'
    , 'datamanagement': 'lsstcorp'
    , 'mule': 'mulesoft'
    , 'mulestudio': 'mulesoft'
    , 'springxd': 'spring'
    , 'talenddataquality': 'talendforge'
    , 'talendesb': 'talendforge'
}

for data in datasetDict.keys():
    subprocess.call(['wget', "https://github.com/hoangdzung/datasets/blob/master/storypoint/IEEE%20TSE2018/Deep-SE/data/{}.pkl.gz".format(data)])

for vocab in set(datasetDict.values()):
    subprocess.call(['wget', "https://github.com/hoangdzung/datasets/blob/master/storypoint/IEEE%20TSE2018/Deep-SE/data/{}.dict.pkl.gz".format(vocab)])

