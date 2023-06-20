from pytz import utc
from datetime import datetime,  timezone

events_name1={
    'Remove bolt':datetime(2022,4,25,8,5,0,tzinfo=utc),
  #  'Incomplete (?) Bolt':datetime(2022,4,27,8,57,0,tzinfo=utc),
    'Loose bolt':datetime(2022,4,29,9,22,0,tzinfo=utc),
    'Tighten bolt':datetime(2022,5,3,8,35,0,tzinfo=utc),
    'Buckling EW916':datetime(2022,5,9,8,5,0,tzinfo=utc),
    'Removal EW916':datetime(2022,5,16,7,36,0,tzinfo=utc),
    'Removal EW918':datetime(2022,5,23,12,4,0,tzinfo=utc),
    'Replacement EW916/918':datetime(2022,5,30,6,51,tzinfo=utc),
    'Grinding EW196':datetime(2022,6,7,8,9,tzinfo=utc),
  #  'climbing start':datetime(2022,6,7,7,59,tzinfo=utc),
  #  'climbing end':datetime(2022,6,7,8,6,tzinfo=utc),        
    'Reinforcement EW916 ':datetime(2022,6,13,8,6,tzinfo=utc), # Exact hour not documented
    'Power line work' :datetime(2022,6,20,20,tzinfo=utc)
}
events_name2={
    'Remove bolt':'2022-04-25',
    'Loose bolt':'2022-04-29',
    'Tighten bolt':'2022-05-03',
    'Buckling EW916':'2022-05-09',
    'Removal EW916':'2022-05-16',
    'Removal EW918':'2022-05-23',
    'Replacement EW916/918':'2022-05-30',
    'Grinding EW196':'2022-06-07',
    'Reinforcement EW916 ':'2022-06-13',
    'Power line work':'2022-06-20'
}
events_chr = {chr(i+97):v for i,v in enumerate(events_name1.keys())}