import os
import glob
import sys
import shutil
# import exifread
import time
import datetime
import operator

QING_EXIF_DATETIME = 'Image DateTime'
QING_TIME_STAMP = datetime.datetime(1990, 8, 22, 0, 0, 0)
QING_CLASSIFIED_INTERVAL = 5

# isvalue = 0: sort on keys
# isvalue = 1: sort on values
# result is a tuple


def qing_sort_dict(mydict, isvalue):
    sorted_mydict = sorted(mydict.items(), key=operator.itemgetter(isvalue))
    return sorted_mydict


def qing_zero_if_none(x):
    return 0 if x is None else x

# records = dict( ( k, 0 if v is None else v ) for k, v in records.items() )
# records = dict( ( k, qing_zero_if_none( records[k] ) )  for k in records)


def qing_str_to_time(timestr):
    t = time.strptime(timestr, '%Y:%m:%d %H:%M:%S')
    return t


def qing_time_to_datetime(t):
    d = datetime.datetime(* t[:6])
    return d


def qing_str_to_datetime(timestr):
    dtime = qing_str_to_time(timestr)
    return qing_time_to_datetime(dtime)


def qing_datetime_diff_seconds(dtime):
    return (dtime - QING_TIME_STAMP).seconds


def qing_traverse_dict(records):
    for key, value in records.iteritems():
        print(key, value)


# def qing_show_exif(path_name):
#     f = open(path_name, 'rb')
#     # a dictonary mapping names of exif tages to their values
#     tags = exifread.process_file(f)
#     for tag in tags.keys():
#         if tag in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
#             print("Key: %s, value %s" % (tag, tags[tag]))


# def qing_read_exif(path_name, key):
#     f = open(path_name, 'rb')
#     tags = exifread.process_file(f)
#     return tags[key]


def qing_mkdir(ndir):
    if not os.path.exists(ndir):
        os.makedirs(ndir)
    pass


def qing_check_substr(ss, sub):
    if sub in ss:
        return True
    return False

# copy all file_typed files from old_path to new_path


def qing_cp_files(old_path, new_path, file_type):
    qing_mkdir(new_path)
    # print(old_path + '/*.' + file_type)
    files = sorted(glob.glob(old_path + '/*.' + file_type))
    for f in files:
        filename = os.path.basename(f)
        if qing_check_substr(filename, 'FRM'):
            new_f = new_path + '/' + filename
            print('copy ' + f + ' ' + new_f)
            shutil.copy(f, new_f)


def qing_mv_files(old_path, new_path, file_type):
    qing_mkdir(new_path)
    # print(old_path + '/*.' + file_type)
    files = sorted(glob.glob(old_path + '/*.' + file_type))
    for f in files:
        filename = os.path.basename(f)
        if qing_check_substr(filename, 'FRM'):
            new_f = new_path + '/' + filename
            print('move ' + f + ' ' + new_f)
            shutil.move(f, new_f)


def qing_imagelist_generator(out_filename, dir_path, filetype):
    out_file_object = open(out_filename, 'w')
    files = sorted(glob.glob(dir_path + '/*.' + filetype))
    for f in files:
        filename = os.path.basename(f)
        out_file_object.write(filename + '\n')
    out_file_object.close()


def main():
    pass


if __name__ == '__main__':
    main()
