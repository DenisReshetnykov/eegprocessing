from array import array
import numpy as np

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

import pyedflib
from PIL import Image, ImageFont, ImageDraw

import sys
import scipy.stats as st

import time

# import pyeeg

# Read EDF file and print some usefull data
def read_edf_file(file):
    f = pyedflib.EdfReader(file)
    if False:
        print("signallabels: %s" % f.getSignalLabels())
        print("singnals in file: %s" % f.signals_in_file)
        print("datarecord duration: %f seconds" % f.getFileDuration())
        print("samplefrequency: %f" % f.getSampleFrequency(1))
        print("read header: " + str(f.getPatientName()))
        print("read anotations: " + str(f.readAnnotations()))
        # print("samplefrequency: %f" % f.getSampleFrequency(27))
        # f._close()
    return f


def get_data_from_chanel(channel, file, filetype='edf', around=True):
    if filetype == 'edf':
        f = read_edf_file(file)
    data = f.readSignal(channel)
    if around:
        data = np.around(data)
    f._close()
    return data


def get_lead_name_by_channel(channel, file, filetype='edf'):
    if filetype == 'edf':
        f = read_edf_file(file)
        lead_name = f.getLabel(channel)
    f._close()
    return lead_name


def create_file_dictionary(folder_path):
    eeg_file_dict = {}
    if os.path.exists(folder_path):
        file_list = os.listdir(folder_path)
        for entry in file_list:
            eeg_file_dict[ entry.split('c')[-1].split('p')[0] ] = []
        for entry in file_list:
            eeg_file_dict[ entry.split('c')[-1].split('p')[0] ].append(entry)
# create_file_dictionary('testdata/EGE/')


def stat_scratch(channels, file):
    f = read_edf_file(file)
    i = 0
    for channel in channels:
        i += 1
        mpl.rcParams['axes.titlesize'] = 'small'
        _x = f.readSignal(channel)
        lead = f.getLabel(channel)
        if False:
            result = []
            result.append(len(_x))  # Число элементов выборки
            result.append(np.mean(_x))  # среднее
            result.append((np.min(_x), np.max(_x)))  # (min, max)
            result.append(np.std(_x))  # стандартное отклонение
            result.append(100.0 * result[-1] / result[0])  # коэффициент вариации (Пирсона)
            result.append((np.percentile(_x, 25), np.percentile(_x, 50), np.percentile(_x, 75)))  # квартили
            result.append(st.mode(_x))  # мода
            result.append(st.skew(_x))  # асимметрия
            result.append(st.kurtosis(_x))  # эксцесс
            _range = np.linspace(0.9 * np.min(_x), 1.1 * np.max(_x), 100)  # область определения для оценки плотности
            result.append((_range, st.gaussian_kde(_x)(_range)))  # оценка плотности распределения

            # Вычисление важных показателей
            n, m, minmax, s, cv, perct, mode, skew, kurt, kde = tuple(result)
            print('Число элементов выборки: {0:d}'.format(n))
            print('Среднее значение: {0:.4f}'.format(m))
            print('Минимальное и максимальное значения: ({0:.4f}, {1:.4f})'.format(*minmax))
            print('Стандартное отклонение: {0:.4f}'.format(s))
            # print('Коэффициент вариации (Пирсона): {0:.4f}'.format(cv))
            print('Квартили: (25%) = {0:.4f}, (50%) = {1:.4f}, (75%) = {2:.4f}'.format(*perct))
            print('Коэффициент асимметрии: {0:.4f}'.format(skew))
            print('Коэффициент эксцесса: {0:.4f}'.format(kurt))
            # print('оценка плотности распределения: {0:.4f}'.format(kde))

    _x_around = np.around(_x)
    f._close()
    return _x_around, lead


def local_rank_coding(data, rank_quantity):
    coded_data = np.zeros(len(data))
    min = np.min(data)
    max = np.max(data)
    bandwidth = (max - min) / rank_quantity
    for i in range(len(data)):
        coded_data[i] = np.floor((data[i]-min)/bandwidth)
    return coded_data


def shanon_entropy(data, window_size):
    ent = np.zeros(len(data) - window_size)
    for i in range(len(data) - window_size):
        data_chunk = data[0+i:window_size+i]
        # Create a frequency data
        freq_list = []
        for entry in set(data_chunk):
            counter = 0
            for j in data_chunk:
                if j == entry:
                    counter += 1
            freq_list.append(float(counter) / window_size)
        # Shannon entropy
        for freq in freq_list:
            ent[i] += freq * np.log2(freq)
    ent = -ent
    return ent


def full_entropy(data):
    entr=0
    freq_list = []
    for entry in set(data):
        counter = 0
        for j in data:
            if j == entry:
                counter += 1
        freq_list.append(float(counter) / len(data))
    for freq in freq_list:
        entr += freq * np.log2(freq)
    entr = -entr
    return entr


def unconditional_entropy(data, chain_size):
    """
    :param data: 1D array of byte data
    :param chain_size: lengs of symbol chain for conditional entropy
    :return: unconditional entropy E(chain_size)
    """
    data_array = np.zeros((len(data)-chain_size+1,chain_size),dtype='b')
    for index in range(len(data)-chain_size+1):
        data_array[index]=data[index:index+chain_size]
    freq_list = []
    for entry in set(tuple(i) for i in data_array):
        counter = 0
        for element in data_array:
            if (entry == element).all():
                counter += 1
        freq_list.append(float(counter) / len(data_array))
    ent = 0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent
    return ent


#first derivative of the table function
def table_first_derivative(data):
    data_d_left = np.zeros(len(data))
    data_d_right = np.zeros(len(data))
    data_d_center = np.zeros(len(data))
    for j in range(len(data)):
        if j == 0:
            data_d_right[j] = data[j + 1] - data[j]
            data_d_left[j] = data_d_right[j]
            data_d_center[j] = data_d_right[j]
        elif j == len(data)-1:
            data_d_left[j] = data[j] - data[j - 1]
            data_d_right[j] = data_d_left[j]
            data_d_center[j] = data_d_left[j]
        else:
            data_d_left[j] = data[j] - data[j - 1]
            data_d_right[j] = data[j + 1] - data[j]
            data_d_center[j] = (data[j + 1] - data[j - 1])/2
    return data_d_left, data_d_right, data_d_center

#second derivative of the table function
def table_first_derivative(data):
    data_d_left = np.zeros(len(data))
    data_d_right = np.zeros(len(data))
    data_d_center = np.zeros(len(data))
    for j in range(len(data)):
        if j == 0:
            data_d_right[j] = data[j + 1] - data[j]
            data_d_left[j] = data_d_right[j]
            data_d_center[j] = data_d_right[j]
        elif j == len(data)-1:
            data_d_left[j] = data[j] - data[j - 1]
            data_d_right[j] = data_d_left[j]
            data_d_center[j] = data_d_left[j]
        else:
            data_d_left[j] = data[j] - data[j - 1]
            data_d_right[j] = data[j + 1] - data[j]
            data_d_center[j] = (data[j + 1] - data[j - 1])/2
    return data_d_left, data_d_right, data_d_center

# Надо исправить нулевые значения для краев выборки
def avg_fiter(data, filter_window_range, filter_count):
    filtered_data = np.zeros(len(data))
    for n in range(filter_count):
        for i in range(len(data)):
            filtered_data[i]=data[i]
            if (i>=filter_window_range) and (i<len(data)-filter_window_range):
                for j in range(filter_window_range):
                    filtered_data[i]=filtered_data[i]+data[i-(j+1)]+data[i+(j+1)]
            elif i<filter_window_range:
                for j in range(filter_window_range):
                    filtered_data[i]=filtered_data[i]+data[i+(j+1)]
            else:
                for j in range(filter_window_range):
                    filtered_data[i]=filtered_data[i]+data[i-(j+1)]
            filtered_data[i] = filtered_data[i] / (2 * filter_window_range + 1)
        for i in range(len(data)):
            data[i] = filtered_data[i]
    return filtered_data


#Параметры вызова
window_size = 250
eeg_file_dict = {1:['1p11','1p11f','c1p11'], 2:['2p11','2p11f','c2p11'], 3:['3p11','3p11f','c3p11a'], 5:['5p11','5p11f','c5p11a'], 6:['6p11','6p11f','c6p11a']}
eeg_file_dict2 = {5:['5p11','5p11f','c5p11a'], 6:['6p11','6p11f','c6p11a']}
rank_quantity = 8
filter_window_size = 7
filter_count = 2
folder_name = 'plotresults/'+str(window_size)+'-'+str(rank_quantity)+'/'+str(filter_window_size)+'/'+str(filter_count)+'/'
folder_name_for_cond_entropy = 'plotresults/conditional/'+'rank = '+str(rank_quantity)+'/filter_size = '+str(filter_window_size)+'/filter_count = '+str(filter_count)+'/'
folder_name_for_aproximate_entropy = 'plotresults/aproximate/'+'filter_size = '+str(filter_window_size)+'/filter_count = '+str(filter_count)+'/'

eeg = get_data_from_chanel(1, "testdata/EGE/" + eeg_file_dict[1][0] + ".edf")
lead_eeg = get_lead_name_by_channel(1, "testdata/EGE/" + eeg_file_dict[1][0] + ".edf")
# clear_data = avg_fiter(eeg, filter_window_size, filter_count)
# coded_data = local_rank_coding(clear_data, rank_quantity)

def conditional_entropy(data, min=1,max=10):
    uncond_ent = np.zeros(max-min+2)
    cond_ent = np.zeros(max-min+1)
    for i in range(min-1,max+1):
        uncond_ent[i+1-min] = unconditional_entropy(data,i)
        print("for L = " + str(i) + " unconditional entropy = " + str(uncond_ent[i]))
    for i in range(min,max+1):
        cond_ent[i-min] = uncond_ent[i-min+1]-uncond_ent[i-min]
        print("for L = "+str(i)+" conditional entropy = "+ str(cond_ent[i-1]))
    return cond_ent

def theta_entropy(data, m, r=None, phase_width=None):
    data_array = np.zeros((len(data) - m + 1, m))
    C = np.zeros(len(data_array))

    if phase_width is None:
        phase_width = 0.1

    if r is None:
        r = phase_width*np.std(data)
        print('r = '+str(r))

    for index in range(len(data) - m + 1):
        data_array[index] = data[index:index + m]

    for i in range(len(data_array)):
        for j in range(i,len(data_array)):
            if max(abs(data_array[i]-data_array[j]))<=r :
                C[i] += 1 / (len(data_array))
                C[j] += 1 / (len(data_array))
        # if i%1000 == 0:
        #     print(str(i)+' arrived')

    theta = np.sum(np.log(C))/len(data_array)
    return theta


def aproximate_entropy(data, m_min=1, m_max=6, r=None, phase_width=None):
    aprox_ent = np.zeros(m_max-m_min+1)
    theta = np.zeros(m_max-m_min+2)
    for i in range(m_min,m_max+2):
        theta[i-m_min] = theta_entropy(data, i, r, phase_width)
        # print('for m = '+str(i)+' theta entropy = '+str(theta[i-m_min]))
    for i in range(m_min,m_max+1):
        aprox_ent[i-m_min] = theta[i-m_min]-theta[i-m_min+1]
        # print('for m = '+str(i)+' aproximate entropy = '+str(theta[i-m_min]))
    return aprox_ent


def create_signal(l):
    signal = [3*np.cos(np.pi*i/20) for i in range(l)]
    return signal


def create_random_signal(l):
    r_signal = 2 + 0.3*np.random.randn(l)
    return r_signal


def plot_aproximate_entropy(eeg_file_dict):
    if not os.path.exists(folder_name_for_aproximate_entropy):
        os.makedirs(folder_name_for_aproximate_entropy)
    for key in eeg_file_dict.keys():
        for chanel in range(10):
            eeg = get_data_from_chanel(1, "testdata/EGE/" + eeg_file_dict[key][0] + ".edf")
            lead_eeg = get_lead_name_by_channel(1, "testdata/EGE/" + eeg_file_dict[key][0] + ".edf")
            aprox_entropy = aproximate_entropy(avg_fiter(eeg, filter_window_size, filter_count), 1, 6, phase_width=0.1)

            eeg_f, lead_eeg_f = stat_scratch([chanel], "testdata/EGE/" + eeg_file_dict[key][1] + ".edf")
            aprox_entropy_f = aproximate_entropy(avg_fiter(eeg_f, filter_window_size, filter_count), 1, 6, phase_width=0.1)

            eeg_c, lead_eeg_c = stat_scratch([chanel], "testdata/EGE/" + eeg_file_dict[key][2] + ".edf")
            aprox_entropy_c = aproximate_entropy(avg_fiter(eeg_c, filter_window_size, filter_count), 1, 6, phase_width=0.1)

            fig, ax1 = plt.subplots()
            fig.set_size_inches(5, 5)

            ax1.set_xticklabels([i+1 for i in range(6)])
            ax1.plot(aprox_entropy, linewidth=0.8, color='green')
            ax1.plot(aprox_entropy_f, linewidth=0.8, color='red')
            ax1.plot(aprox_entropy_c, linewidth=0.8, color='blue')

            ax1.set_title('Испытуемый ' + eeg_file_dict[key][0][0] + ' отведение ' + lead_eeg[4:], fontsize=15)
            ax1.set_ylabel('ApEn(m)', color='black')
            ax1.set_xlabel('m', color='black')
            ax1.grid(b=True, linewidth=0.5, color='black', linestyle='--')
            ax1.legend(['Спецефическая', 'Неспецефическая', 'Контроль'])

            # ax1.set_ylim(-40, 160)

            fig.tight_layout()
            print('exam ' + eeg_file_dict[key][0][0] + ' chanel ' + str(chanel) + ' done')
            plt.savefig(folder_name_for_aproximate_entropy + 'ex' + eeg_file_dict[key][0][0] + lead_eeg[4:] + '.png', format='png')
            text_file = open(folder_name_for_aproximate_entropy + 'ex' + eeg_file_dict[key][0][0] + lead_eeg[4:] + '.txt', 'w')
            text_file.write(str(aprox_entropy)+'\n'+str(aprox_entropy_f)+'\n'+str(aprox_entropy_c))
            text_file.close()
            plt.clf()

# plot_aproximate_entropy(eeg_file_dict2)


def plot_cond_entropy(eeg_file_dict):
    if not os.path.exists(folder_name_for_cond_entropy):
        os.makedirs(folder_name_for_cond_entropy)
    for key in eeg_file_dict.keys():
        for chanel in range(10):
            eeg, lead_eeg = stat_scratch([chanel], "testdata/EGE/" + eeg_file_dict[key][0] + ".edf")
            cond_entropy = conditional_entropy(local_rank_coding(avg_fiter(eeg, filter_window_size, filter_count), rank_quantity), 1, 10)

            eeg_f, lead_eeg_f = stat_scratch([chanel], "testdata/EGE/" + eeg_file_dict[key][1] + ".edf")
            cond_entropy_f = conditional_entropy(local_rank_coding(avg_fiter(eeg, filter_window_size, filter_count), rank_quantity), 1, 10)

            eeg_c, lead_eeg_c = stat_scratch([chanel], "testdata/EGE/" + eeg_file_dict[key][2] + ".edf")
            cond_entropy_c = conditional_entropy(local_rank_coding(avg_fiter(eeg, filter_window_size, filter_count), rank_quantity), 1, 10)

            fig, ax1 = plt.subplots()
            ax1.set_title('Испытуемый ' + eeg_file_dict[key][0][0] + ' отведение ' + lead_eeg[4:], fontsize=15)
            fig.set_size_inches(5, 5)

            ax1.plot(cond_entropy, linewidth=0.8, color='green')
            ax1.plot(cond_entropy_f, linewidth=0.8, color='red')
            ax1.plot(cond_entropy_c, linewidth=0.8, color='blue')
            ax1.set_xticklabels([i + 1 for i in range(10)])
            ax1.set_ylabel('conditional entropy', color='black')
            # ax1.set_ylim(-40, 160)

            fig.tight_layout()
            print('exam ' + eeg_file_dict[key][0][0] + ' chanel ' + str(chanel) + ' done')
            plt.savefig(folder_name_for_cond_entropy + 'ex' + eeg_file_dict[key][0][0] + lead_eeg[4:] + '.png', format='png')
            text_file = open(folder_name_for_cond_entropy + 'ex' + eeg_file_dict[key][0][0] + lead_eeg[4:] + '.txt', 'w')
            text_file.write(str(cond_entropy)+'\n'+str(cond_entropy_f)+'\n'+str(cond_entropy_c))
            text_file.close()
            plt.clf()

# plot_cond_entropy(eeg_file_dict)


# timer = np.zeros(3)
# timer[0] = time.time()
# print('start = '+str(time.ctime()))
# aproximate_entropy(clear_data[10000:14999], 2, 2)
# print('after my = '+str(time.ctime()))
# timer[1] = time.time()
# print('proshlo = '+str(timer[1]-timer[0]))
# ApEn(clear_data[10000:14999], 2, 1.49603213429)
# print('after not my = '+str(time.ctime()))
# timer[2] = time.time()
# print('proshlo = '+str(timer[2]-timer[1]))



def plot_eeg_entropy():
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for key in eeg_file_dict.keys():
        for chanel in range(10):
            eeg, lead_eeg = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][0]+".edf")
            ent = shanon_entropy(local_rank_coding(avg_fiter(eeg, filter_window_size, filter_count), rank_quantity), window_size)
            # plt.plot(ent, linewidth=0.2, color = 'green')

            eeg_f, lead_eeg_f = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][1]+".edf")
            ent_f = shanon_entropy(local_rank_coding(avg_fiter(eeg_f, filter_window_size, filter_count), rank_quantity), window_size)
            # plt.plot(ent_f, linewidth=0.2, color = 'red')

            eeg_c, lead_eeg_c = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][2]+".edf")
            ent_c = shanon_entropy(local_rank_coding(avg_fiter(eeg_c, filter_window_size, filter_count), rank_quantity), window_size)
            # plt.plot(ent_c, linewidth=0.2, color = 'blue')

            fig, ax1 = plt.subplots()
            ax1.set_title('Испытуемый '+eeg_file_dict[key][0][0]+' отведение '+lead_eeg[4:], fontsize=15)
            fig.set_size_inches(15, 5)


            ax1.plot(np.hstack((eeg_c,eeg,eeg_f)), linewidth=0.2, color = 'black')
            ax1.set_ylabel('µV', color='black')
            ax1.set_ylim(-40, 160)

            ax1.text(0.15, 0.95, 'Контроль', style='italic', transform=ax1.transAxes, fontsize=15)
            ax1.text(0.4, 0.95, 'Спецефическая задача', style='italic', transform=ax1.transAxes, fontsize=15)
            ax1.text(0.7, 0.95, 'Неспецефическая задача', style='italic', transform=ax1.transAxes, fontsize=15)
            ax1.axvline(x=30000, color='red', linestyle='--', linewidth=4)
            ax1.axvline(x=60000, color='red', linestyle='--', linewidth=4)

            ax2 = ax1.twinx()
            ax2.plot(np.hstack((ent_c, ent, ent_f)), linewidth=0.4, color='green')
            ax2.set_ylabel('Энтропия Шеннона, окно '+str(window_size)+' отсчетов', color='green')
            ax2.tick_params('y', colors='green')

        # ent_d_l, ent_d_r, ent_d_c = table_first_derivative(ent)
        # plt.subplot(121)
        # plt.plot(eeg, linewidth=0.2, color = 'black')


        # plt.plot(ent_d_l, color = 'red')
        # plt.plot(ent_d_r, color = 'green')
        # plt.plot(ent_d_c, color = 'blue')

        fig.tight_layout()
        print('exam '+eeg_file_dict[key][0][0]+' chanel '+str(chanel)+' done')
        plt.savefig(folder_name+'ex'+eeg_file_dict[key][0][0]+lead_eeg[4:]+'.png', format='png')
        plt.clf()
        # plt.show()

def scrapping(file):
    f = open(file, 'r')
    s = f.readlines()
    buffer = np.zeros((5,10,3,10))
    for exam in range(5):
        for chanel in range (10):
            for type in range (3):
                for l in range(10):
                    buffer[exam,chanel,type,l] = float(s[exam*640+chanel*64+type*21+11+l].split('=')[2][1:7])
    # buffer_file = open('buffer.txt','w')
    print(buffer[1,1,1])
    return buffer


def print_entropy(buffer):
    f = pyedflib.EdfReader("testdata/EGE/" + eeg_file_dict[1][0] + ".edf")
    for exam in range(5):
        fig, axes = plt.subplots(1, 10, sharey=True)
        fig.set_size_inches(20, 5)
        for chanel in range (10):
            axes[chanel].set_title(f.getSignalLabels()[chanel][4:], fontsize=10)
            axes[chanel].grid(b=True, linewidth=0.5, color='black', linestyle='--' )
            axes[chanel].set_xticks([i for i in range(9)])
            axes[chanel].set_xticklabels([i+2 for i in range(9)])
            axes[chanel].set_ylim(0.1, 0.5)
            axes[chanel].set_xlabel("L")
            axes[chanel].plot(buffer[exam,chanel,0][1:], linewidth=1, color='green', linestyle='-')
            axes[chanel].plot(buffer[exam,chanel,1][1:], linewidth=1, color='red', linestyle='-')
            axes[chanel].plot(buffer[exam,chanel,2][1:], linewidth=1, color='blue', linestyle='-')
        axes[0].set_ylabel("E( L \ L-1 )")
        axes[9].legend(['Спецефическая','Неспецефическая','Контроль'])
        fig.tight_layout()
        plt.savefig('plotresults/'+str(exam)+'.png', format='png')
        plt.clf()
# print_entropy(scrapping('eeg.txt'))


def print_entropy_ras(buffer):
    f = pyedflib.EdfReader("testdata/EGE/" + eeg_file_dict[1][0] + ".edf")
    fig, axes = plt.subplots(1, 10, sharey=True)
    fig.set_size_inches(20, 10)
    for chanel in range(10):
        axes[chanel].set_title(f.getSignalLabels()[chanel][4:], fontsize=10)
        axes[chanel].set_xticks([i for i in range(9)])
        axes[chanel].set_xticklabels([i + 2 for i in range(9)])
        axes[chanel].set_ylim(0.1, 0.5)
        axes[chanel].grid(b=True, linewidth=0.5, color='black', linestyle='--')
        print(buffer[1, chanel, 0][1:])
        print([i for i in range(9)])
        for exam in range(5):
            axes[chanel].plot(buffer[exam, chanel, 0][1:], 'o', color='green')
            axes[chanel].plot(buffer[exam, chanel, 1][1:], 'o', color='red')
            axes[chanel].plot(buffer[exam, chanel, 2][1:], 'o', color='blue')
    axes[0].set_ylabel("E( L \ L-1 )")
    axes[9].legend(['Спецефическая', 'Неспецефическая', 'Контроль'])
    fig.tight_layout()
    plt.savefig('plotresults/r/' + str("1") + '.png', format='png')
    plt.show()
# print_entropy_ras(scrapping('eeg.txt'))


def draw_on_image():
    for key in eeg_file_dict.keys():
        for chanel in range(10):
            eeg, lead_eeg = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][0]+".edf")
            entr = full_entropy(local_rank_coding(avg_fiter(eeg, filter_window_size, filter_count), rank_quantity))
            im = Image.open(folder_name + 'ex' + eeg_file_dict[key][0][0] + lead_eeg[4:] + '.png')
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype("Vera.ttf", 20)
            draw.text((700, 70), "E = "+str(round(entr,2)), (0, 0, 0), font=font)

            eeg_f, lead_eeg_f = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][1]+".edf")
            entr_f = full_entropy(local_rank_coding(avg_fiter(eeg_f, filter_window_size, filter_count), rank_quantity))
            draw.text((1100, 70), "E = "+str(round(entr_f,2)), (0, 0, 0), font=font)

            eeg_c, lead_eeg_c = stat_scratch([chanel], "testdata/"+eeg_file_dict[key][2]+".edf")
            entr_c = full_entropy(local_rank_coding(avg_fiter(eeg_c, filter_window_size, filter_count), rank_quantity))
            draw.text((280, 70), "E = "+str(round(entr_c,2)), (0, 0, 0), font=font)
            im.save(folder_name+'eex'+eeg_file_dict[key][0][0]+lead_eeg[4:]+'.png')
            print('exam ' + eeg_file_dict[key][0][0] + ' chanel ' + str(chanel) + ' done')


def spectral_analyzis(Signal, SignalFreq, Band, EpochStart=0, EpochStop=None, DFreq=1):
    '''
    :param Signal: list, 1-D real signal
    :param EpochStart: integer,
    :param EpochStop: integer,
    :param SignalFreq: integer, Signal physical frequency
    :param Band: list, real frequencies (in Hz) of bins  Each element of Band is a physical frequency and shall not exceed the Nyquist frequency, i.e., half of sampling frequency.
    :param DFreq: integer, number of equal segments in each band
    :return: Power: list, 2-D power in each Band divided on equal DFreq segments
    :return: PowerRatio: spectral power in each segment normalized by total power in ALL frequency bins.
    :return: PowerFreq: Frequencies in wich Power computed
    '''
    SignalSection = Signal[EpochStart:EpochStop]
    fftSignal = abs(np.fft.fft(SignalSection))
    Power = np.zeros((len(Band) - 1) * DFreq)
    PowerFreq = np.zeros((len(Band)-1)*DFreq + 1)

    for BandIndex in range(0, len(Band) - 1):
        Freq = float(Band[BandIndex])
        NextFreq = float(Band[BandIndex + 1])
        for FreqDiff in range(1, DFreq + 1):
            FreqD = Freq + (NextFreq-Freq)*((FreqDiff-1)/DFreq)
            NextFreqD = Freq + (NextFreq-Freq)*(FreqDiff/DFreq)
            Power[BandIndex * DFreq + FreqDiff - 1] = sum(fftSignal[int(
                np.floor(FreqD / SignalFreq * len(SignalSection))): int(
                np.floor(NextFreqD / SignalFreq * len(SignalSection)))])
            PowerFreq[BandIndex * DFreq + FreqDiff - 1] = FreqD
    PowerRatio = Power / sum(Power)
    PowerFreq[-1]=Band[-1]
    return Power, PowerRatio, PowerFreq


# stat_scratch([chanel], "testdata/"+eeg_file_dict[1][0]+".edf")
# Power, PowerRatio, PowerFreq = spectral_analyzis(eeg, 500, [0.5, 4, 8, 13, 40], 0, 30000, 1)
# print("Power = "+str(Power))
# print("PowerRatio = "+str(PowerRatio))
# print("Sum" + str(sum(PowerRatio)))
# print("PowerFreq= "+str(PowerFreq))

def spectral_window(signal, Band, SignalFreq, windowSize=512, windowShift=1):
    '''
    :param signal:
    :param Band:
    :param SignalFreq:
    :param windowSize:
    :param windowShift:
    :return:
    '''
    l = len(signal)
    Power = np.zeros((l//windowShift, len(Band)-1))
    PowerRatio = np.zeros((l//windowShift, len(Band)-1))
    PowerFreq = np.zeros(len(Band))
    iter = 0
    for i in range(0, l-windowSize, windowShift):
        Power[iter], PowerRatio[iter], PowerFreq = spectral_analyzis(signal, SignalFreq, Band, i, i+windowSize)
        # print('for i = '+str(i)+' PowerRatio = '+ str(PowerRatio[iter]))
        iter += 1
    return Power, PowerRatio

Power, PowerRatio = spectral_window(eeg, [0.5, 4, 8, 13, 40], 500, windowSize=1024, windowShift=128)

def plot_spectral_window(data):
    fig, ax1 = plt.subplots()
    # ax1.set_title('Испытуемый ' + eeg_file_dict[key][0][0] + ' отведение ' + lead_eeg[4:], fontsize=15)

    ax1.fill_between(range(len(data)),
                     [data[i][0] for i in range(len(data))], y2=0,
                     color='green', linewidth=0.5, alpha=0.3)
    ax1.fill_between(range(len(data)),
                     [data[i][0]+data[i][1] for i in range(len(data))], [data[i][0] for i in range(len(data))],
                     color='blue', linewidth=0.5, alpha=0.3)
    ax1.fill_between(range(len(data)),
                     [data[i][0]+data[i][1]+data[i][2] for i in range(len(data))], [data[i][0]+data[i][1] for i in range(len(data))],
                     color='yellow', linewidth=0.5, alpha=0.3)
    ax1.fill_between(range(len(data)),
                     [data[i][0]+data[i][1]+data[i][2]+data[i][3] for i in range(len(data))], [data[i][0]+data[i][1]+data[i][2] for i in range(len(data))],
                     color='red', linewidth=0.5, alpha=0.3)
    plt.show()


plot_spectral_window(PowerRatio)

