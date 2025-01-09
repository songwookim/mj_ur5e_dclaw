import matplotlib.pyplot as plt
import numpy as np

# Interactive mode 활성화
plt.ion()

# 초기 설정
fig, ax = plt.subplots()

x = []  # x 데이터를 저장할 리스트 (time)
y1 = []  # 첫 번째 y 데이터 저장
y2 = []  # 두 번째 y 데이터 저장

line1, = ax.plot([], [], label="sin(x)")  # 첫 번째 선
line2, = ax.plot([], [], label="cos(x)")  # 두 번째 선

plt.title("Time-varying Multiple Lines Plot")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# x축과 y축 초기 범위 설정
ax.set_xlim(0, 10)  # 초기 x축 범위
ax.set_ylim(-1.5, 1.5)  # 초기 y축 범위

# 데이터 업데이트 루프
for t in range(200):  # 200번 반복
    # 새로운 데이터 추가
    new_x = t * 0.1  # time에 비례하는 새로운 x값
    new_y1 = np.sin(new_x)  # y1 = sin(x)
    new_y2 = np.cos(new_x)  # y2 = cos(x)
    
    x.append(new_x)
    y1.append(new_y1)
    y2.append(new_y2)
    
    # 데이터를 업데이트
    line1.set_xdata(x)
    line1.set_ydata(y1)
    line2.set_xdata(x)
    line2.set_ydata(y2)
    
    # x축 범위를 동적으로 업데이트
    if new_x > ax.get_xlim()[1]:  # x축 최대값을 초과하면
        ax.set_xlim(ax.get_xlim()[0], new_x + 1)  # 범위를 확장
    
    # 그래프 갱신
    plt.pause(0.1)

plt.ioff()  # Interactive mode 비활성화
plt.show()
