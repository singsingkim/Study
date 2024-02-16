# ★ 시각화 ★
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
font_path = "c:\Windows\Fonts\MALGUN.TTF"
font_name=fm.FontProperties(fname=font_path).get_name()
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.title('캘리포니아 로스')        # 한글깨짐 해결할것
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
