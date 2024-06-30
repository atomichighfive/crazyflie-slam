import argparse
import time
import socket, os, struct, time
import numpy as np
import cv2
import logging
from pathlib import Path

def main(n, p, save):
  logging.info(f"Connecting to AI Deck at {n}:{p}")
  client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client_socket.connect((n, p))
  logging.info(f"Connected")

  imgdata = None
  data_buffer = bytearray()

  if save:
    output_path = Path(f"./output/image-streamer/{time.strftime('%Y%m%d-%H%M%S')}/")
    os.makedirs(output_path, exist_ok=False)
    logging.info(f"Saving output to {output_path}")
  
  count = 0
  while(1):
    packetInfoRaw = read_bytes_from_socket(client_socket, 4)

    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

    imgHeader = read_bytes_from_socket(client_socket, length - 2)

    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic == 0xBC:
      imgStream = bytearray()

      while len(imgStream) < size:
        packetInfoRaw = read_bytes_from_socket(client_socket, 4)
        [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
        chunk = read_bytes_from_socket(client_socket, length - 2)
        imgStream.extend(chunk)

      bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
      bayer_img.shape = (244, 324)
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
  parser.add_argument("-n", metavar="ip", help="AI-deck IP")
  parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
  parser.add_argument('--save', action='store_true', help="Save streamed images")
  args = parser.parse_args()

  main(**vars(args))


