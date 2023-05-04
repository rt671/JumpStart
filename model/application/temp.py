import csv

with open('ratings.dat', 'r') as dat_file:

    with open('ratings.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['UserID', 'MovieID', 'Rating', 'Timestamp'])

        # Loop through each line in the ratings.dat file
        for line in dat_file:
            user_id, movie_id, rating, timestamp = line.strip().split('::')
            writer.writerow([user_id, movie_id, rating, timestamp])
