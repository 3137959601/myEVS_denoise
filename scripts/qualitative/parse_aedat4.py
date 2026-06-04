"""Parse DVSNOISE20 .aedat4 to .npz without dv-processing. Pure Python + lz4."""
import struct, lz4.block, numpy as np, os, sys

def parse_aedat4(path, max_events=None):
    with open(path, 'rb') as f:
        magic = f.read(14)
        assert magic == b'#!AER-DAT4.0\r\n', f'Bad magic: {magic}'
        header_size = struct.unpack('<I', f.read(4))[0]
        num_entries = struct.unpack('<I', f.read(4))[0]
        table = f.read(header_size)

        # Extract XML
        xml_start = table.find(b'<dv')
        xml = table[xml_start:].decode('utf-8', errors='replace')

        # Extract sizeX, sizeY from XML
        import re
        sx = re.search(r'sizeX"[^>]*>(\d+)<', xml)
        sy = re.search(r'sizeY"[^>]*>(\d+)<', xml)
        width = int(sx.group(1)) if sx else 346
        height = int(sy.group(1)) if sy else 260

        data_start = 14 + 4 + 4 + header_size
        f.seek(0, 2)
        total_size = f.tell()

        events = []
        pos = data_start

        while pos < total_size:
            f.seek(pos)
            if pos + 8 > total_size:
                break
            header = f.read(8)
            if len(header) < 8:
                break

            # Packet header: stream_id(2B), packet_type(2B), packet_size(4B)
            sid = struct.unpack('<H', header[0:2])[0]
            ptype = struct.unpack('<H', header[2:4])[0]
            psize = struct.unpack('<I', header[4:8])[0]

            if psize == 0 or psize > total_size - pos:
                pos += 8
                continue

            if sid == 0:  # EVTS (events) stream
                compressed = f.read(psize)
                if len(compressed) < psize:
                    break
                try:
                    raw = lz4.block.decompress(compressed, uncompressed_size=psize*10)
                except:
                    raw = lz4.block.decompress(compressed)

                # Parse events: each event is 8 bytes
                # timestamp(4B) | x(2B) | y(2B) | ? + polarity
                # DV format: [t(4B) | data(4B)] where data = [valid(1B)|pol(1B)|y(2B)|x(2B)]
                # But actual encoding: looking at data...
                # Let's try standard DV encoding:
                # data[0]: valid(1) | polarity(1) | y_hi_bits
                # data[1-2]: x(2B)

                for i in range(0, len(raw), 8):
                    if i + 8 > len(raw):
                        break
                    t = struct.unpack('<I', raw[i:i+4])[0]
                    d = raw[i+4:i+8]
                    # DV event encoding
                    valid = (d[0] >> 7) & 1
                    if not valid:
                        continue
                    pol = (d[0] >> 6) & 1
                    pol_signed = 1 if pol else -1
                    y = ((d[0] & 0x3F) << 6) | ((d[1] >> 2) & 0x3F)
                    x = ((d[1] & 0x3) << 14) | ((d[2] << 6) | (d[3] >> 2))

                    if x < width and y < height:
                        events.append((t, x, y, pol_signed))

                        if max_events and len(events) >= max_events:
                            break
            else:
                f.seek(psize, 1)  # skip non-event data

            pos = f.tell()
            if max_events and len(events) >= max_events:
                break

    if not events:
        raise RuntimeError('No events parsed! Trying alternative encoding...')

    events.sort(key=lambda e: e[0])
    arr = np.array(events, dtype=[('t', '<u8'), ('x', '<i4'), ('y', '<i4'), ('p', '<i1')])
    arr['t'] = arr['t'].astype(np.uint64)

    return arr, width, height


if __name__ == '__main__':
    path = r'D:/hjx_workspace/scientific_reserach/dataset/DVSNOISE20/stairs-2019_10_10_12_58_54.aedat4'
    out = r'D:/hjx_workspace/scientific_reserach/projects/myEVS/data/qualitative/converted/dvsnoise20/stairs-2019_10_10_12_58_54.npz'

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f'Parsing: {path}')
    arr, w, h = parse_aedat4(path)
    print(f'Parsed {len(arr)} events, resolution {w}x{h}')
    print(f'First: t={arr[0][0]} x={arr[0][1]} y={arr[0][2]} p={arr[0][3]}')
    print(f'Last:  t={arr[-1][0]} x={arr[-1][1]} y={arr[-1][2]} p={arr[-1][3]}')

    np.savez_compressed(out, t=arr['t'], x=arr['x'], y=arr['y'], p=arr['p'])
    print(f'Saved to: {out}')
