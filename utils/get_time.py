import time
 
def get_date_time_now():
    time_now = time.ctime()

    time_now = time_now.replace(':', '-')

    return time_now

print(get_date_time_now())