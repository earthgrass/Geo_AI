import os

def read_kong_rey(cma_file="CH2024BST.txt"):
    """提取康妮 (KONG-REY) 原始轨迹"""
    if not os.path.exists(cma_file):
        return None, []
    with open(cma_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = ""
    track_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('66666'):
            parts = line.split()
            n = int(parts[2])
            name = parts[7] if len(parts) > 7 else ""
            if "KONG-REY" in name.upper():
                header = line
                for j in range(1, n+1):
                    track_lines.append(lines[i+j].strip())
                break
            i += n + 1
        else:
            i += 1
    return header, track_lines

def write_virtual_typhoons(header, track_lines, out_file="Virtual_Typhoons_2026.txt"):
    with open(out_file, 'w', encoding='utf-8') as f:
        
        # 1. V-SHIFT: 单纯路径偏移 (North +3.0°, West -3.0°)
        f.write(header.replace("KONG-REY", "V-SHIFT") + "\n")
        for line in track_lines:
            p = line.split()
            lat = int(p[2]) + 30  
            lon = int(p[3]) - 30  
            f.write(f"{p[0]} {p[1]} {lat} {lon} {p[4]} {p[5]}\n")
            
        # 2. V-INTENSE: 单纯热力增强 (Press -15 hPa, Wind * 1.15)
        f.write(header.replace("KONG-REY", "V-INTENSE") + "\n")
        for line in track_lines:
            p = line.split()
            lat = int(p[2]) 
            lon = int(p[3])
            press = max(880, int(p[4]) - 15)
            wind = int(float(p[5]) * 1.15)
            f.write(f"{p[0]} {p[1]} {lat} {lon} {press} {wind}\n")

        # 3. V-COMPOUND: 路径偏移 + 强度增强 (North +3.0, West -3.0, Press -15)
        f.write(header.replace("KONG-REY", "V-COMPOUND") + "\n")
        for line in track_lines:
            p = line.split()
            lat = int(p[2]) + 30
            lon = int(p[3]) - 30
            press = max(880, int(p[4]) - 15)
            wind = int(float(p[5]) * 1.15)
            f.write(f"{p[0]} {p[1]} {lat} {lon} {press} {wind}\n")

        # 4. V-SLOW: 移速减慢滞留 (步长压缩 30%)
        f.write(header.replace("KONG-REY", "V-SLOW") + "\n")
        curr_lat = int(track_lines[0].split()[2])
        curr_lon = int(track_lines[0].split()[3])
        for i, line in enumerate(track_lines):
            p = line.split()
            if i == 0:
                f.write(f"{p[0]} {p[1]} {curr_lat} {curr_lon} {p[4]} {p[5]}\n")
            else:
                prev_p = track_lines[i-1].split()
                delta_lat = (int(p[2]) - int(prev_p[2])) * 0.7 # 移动距离只剩70%
                delta_lon = (int(p[3]) - int(prev_p[3])) * 0.7
                curr_lat += delta_lat
                curr_lon += delta_lon
                f.write(f"{p[0]} {p[1]} {int(curr_lat)} {int(curr_lon)} {p[4]} {p[5]}\n")
            
    print(f"✅ 严谨版《气候变化虚拟台风档案》生成成功: {out_file}")

if __name__ == "__main__":
    h, t = read_kong_rey("CH2024BST.txt")
    if h:
        write_virtual_typhoons(h, t)
    else:
        print("🚨 未找到源数据")