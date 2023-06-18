import time

def calculate_duration(start_time):
  end_time = time.time()
  duration_seconds = end_time - start_time
  duration_hours = duration_seconds / 3600
  duration_rounded = round(duration_hours, 2)

  return duration_rounded