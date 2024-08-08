import argparse
import time
import socket, os, struct, time
import numpy as np
import cv2
import logging
from pathlib import Path
import pandas as pd

def main(n, p, save):
  logging.info(f"Connecting to AI Deck at {n}:{p}")
  client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client_socket.settimeout(10)
  client_socket.connect((n, p))
  logging.info(f"Connected")

  imgdata = None
  data_buffer = bytearray()

  if save:
    output_path = Path(f"./output/image-streamer/{time.strftime('%Y%m%d-%H%M%S')}/")
    os.makedirs(output_path, exist_ok=False)
    logging.info(f"Saving output to {output_path}")
  
  count = 0

  transport_spec = 'HBB'
  image_data_spec = 'BHHBBI'
  # state_spec = 'BfffffffLfffLfffLfff'
  state_spec = 'BfffffffIfffIfffIfff'

  state_df = pd.DataFrame(columns=[
    'state_request_id',
    'attitude_roll',
    'attitude_pitch',
    'attitude_yaw',
    'attitudeQuaternion_x',
    'attitudeQuaternion_y',
    'attitudeQuaternion_z',
    'attitudeQuaternion_w',
    'position_timestamp',
    'position_x',
    'position_y',
    'position_z',
    'velocity_timestamp',
    'velocity_x',
    'velocity_y',
    'velocity_z',
    'acceleration_timestamp',
    'acceleration_x',
    'acceleration_y',
    'acceleration_z'
  ])

  while(1):
    packetInfoRaw = read_bytes_from_socket(client_socket, struct.calcsize('<' + transport_spec))
    [length, routing, function] = struct.unpack('<' + transport_spec, packetInfoRaw)

    imgHeader = read_bytes_from_socket(client_socket, struct.calcsize( '<' + image_data_spec + state_spec))
    [
      magic,
      width,
      height,
      depth,
      format,
      size,
      state_request_id,
      attitude_roll,
      attitude_pitch,
      attitude_yaw,
      attitudeQuaternion_x,
      attitudeQuaternion_y,
      attitudeQuaternion_z,
      attitudeQuaternion_w,
      position_timestamp,
      position_x,
      position_y,
      position_z,
      velocity_timestamp,
      velocity_x,
      velocity_y,
      velocity_z,
      acceleration_timestamp,
      acceleration_x,
      acceleration_y,
      acceleration_z
    ] = struct.unpack('<' + image_data_spec + state_spec, imgHeader)

    if magic == 0xBC:
      dataStream = bytearray()

      while len(dataStream) < size:
        packetInfoRaw = read_bytes_from_socket(client_socket, struct.calcsize('<' + transport_spec))
        [length, dst, src] = struct.unpack('<' + transport_spec, packetInfoRaw)
        chunk = read_bytes_from_socket(client_socket, length-2)
        dataStream.extend(chunk)
        """
        print(
          "---\n"
          f"Transport spec size: {struct.calcsize('<' + transport_spec)}\n"
          f"Length: {length}\n"
          f"Data stream length: {len(dataStream)}\n"
          f"Width: {width}\n"
          f"Height: {height}\n"
          f"Size: {size}"
        )
        """

      new_row = pd.DataFrame.from_records([{
        'state_request_id': state_request_id,
        'attitude_roll': attitude_roll,
        'attitude_pitch': attitude_pitch,
        'attitude_yaw': attitude_yaw,
        'attitudeQuaternion_x': attitudeQuaternion_x,
        'attitudeQuaternion_y': attitudeQuaternion_y,
        'attitudeQuaternion_z': attitudeQuaternion_z,
        'attitudeQuaternion_w': attitudeQuaternion_w,
        'position_timestamp': position_timestamp,
        'position_x': position_x,
        'position_y': position_y,
        'position_z': position_z,
        'velocity_timestamp': velocity_timestamp,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'velocity_z': velocity_z,
        'acceleration_timestamp': acceleration_timestamp,
        'acceleration_x': acceleration_x,
        'acceleration_y': acceleration_y,
        'acceleration_z': acceleration_z
      }])

      state_df = pd.concat([state_df, new_row], ignore_index=True)
      print(state_df.iloc[-1])

      imgStream = dataStream[:]

      bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
      bayer_img.shape = (height, width)
      rgb_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2RGB)
      cv2.imshow('Image', rgb_img)
      if args.save:
        output_file = output_path / f"img_{count:06d}.png"
        cv2.imwrite(output_file, rgb_img)
        logging.info(f"Wrote {output_file}")
      cv2.waitKey(1)
      count += 1


def read_bytes_from_socket(socket, size):
  data = bytearray()
  while len(data) < size:
    data.extend(socket.recv(size-len(data)))
  return data

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
  parser.add_argument("-n", metavar="ip", default='192.168.178.73', help="AI-deck IP")
  parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
  parser.add_argument('--save', action='store_true', help="Save streamed images")
  args = parser.parse_args()

  main(**vars(args))


