def is_room_occupied(occupancy_count):
    return occupancy_count > 0


if __name__ == "__main__":
    count = 2

    if is_room_occupied(count):
        print("Room is OCCUPIED")
    else:
        print("Room is EMPTY")
