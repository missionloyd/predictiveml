import time

def calculate_duration(start_time, precision=2):
  end_time = time.time()
  duration_seconds = end_time - start_time
  duration_hours = duration_seconds / 3600

  hours = int(duration_hours)
  remaining_seconds = duration_seconds - (hours * 3600)
  minutes = int(remaining_seconds / 60)
  seconds = round(remaining_seconds - (minutes * 60), precision)

  return hours, minutes, seconds
