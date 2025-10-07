
panels={'panel-A':("Liliane","Gad"),
        'panel-B':("Miriam","Kaya"),
        'panel-C':("Faith","Sheila"),
        'panel-D':("Faith","Miriam"),
        'panel-E':("Faith","Kaya"),}
dags={"Gad":"265",
      "Liliane":"264",
      "Miriam":"268",
      "Kaya":"263",
      'Faith':"266",
      "Sheila":"267"}

allocation={
    'batch1':'panel-A',
    'batch2':'panel-B',
    'batch3':'panel-C',
    'batch4':'panel-A',
    'batch5':'panel-B',
    'batch6': 'panel-C',
    'batch7': 'panel-B',
    'batch8': 'panel-A',
    'batch9': 'panel-A',
    'batch10': 'panel-A',
    'batch11': 'panel-A',
    'batch12': 'panel-B',
    'batch13': 'panel-B',
    'batch14': 'panel-B',
    'batch15': 'panel-B',
    'batch16': 'panel-C',
    'batch17': 'panel-C',
    'batch18': 'panel-C',
    'batch19': 'panel-C',
    'batch20': 'panel-C',
    'batch21': 'panel-D',
    'batch22': 'panel-D',
    'batch23': 'panel-E',
    'batch24': 'panel-B',
    'batch25': 'panel-B',
    'batch26': 'panel-B',
    'batch27': 'panel-B',
    'batch28': 'panel-E',
    'batch29': 'panel-A',
    'batch30': 'panel-A',
    'batch31': 'panel-A',
    'batch32': 'panel-A',
    'batch33': 'panel-A',
    'batch34': 'panel-B'

}

errors={
    'batch4':[{'id': 61, 'model': 'cn'}, # clinician response dont match vignette
              {'id': 71, 'model': 'cn'},
              {'id':178, 'model': 'cn'}],
    'batch22':[
        {'id':1022, 'model':'cn'},
    ],
    'batch20':[
        {'id':747, 'model':'cn'},
    ],
    'batch11':[
        {'id':256,'model':'cn'},
        {'id':276,'model':'cn'},
        {'id':295, 'model':'cn'},
        {'id':301, 'model':'cn'},
    ],
    'batch30':[
        {'id':4738, 'model':'cn'},
    ],
    'batch31':[
        {'id':5666, 'model':'cn'},
        {'id':6192, 'model':'cn'},
    ],
    'batch32':[
        {'id':6315, 'model':'cn'},
    ],
    'batch33':[
        {'id':6757, 'model':'cn'},
    ],
    'batch34':[
        {'id':8024, 'model':'cn'},
    ]
}