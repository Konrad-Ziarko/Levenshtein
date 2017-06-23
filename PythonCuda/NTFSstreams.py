# -*- coding: UTF-8 -*-

import io, glob, os, sys, tarfile

_ = lambda s: s

import struct

from ctypes import *
from utils import ZlibUtil, sizeof_fmt

class NTFSDataStreams:

    @staticmethod
    def add(path, filename, streamname=None):

        try:
            file = ':'.join([path, streamname or filename])
            with open(file, 'wb') as data, open(filename, 'rb') as streamdata:
                data.write(streamdata.read())

        except IOError as e:
            print(_('Error: {}').format(e.strerror))

    @staticmethod
    def extract(path, streamname, filename=None):

        file = filename or streamname
        stream = ':'.join([path, streamname])

        try:
            with open(file, 'wb') as data, open(stream, 'rb') as streamdata:
                data.write(streamdata.read())

        except IOError as e:
            print(_('Error: {}').format(e.strerror))

    @staticmethod
    def remove(path, streamname):
        stream = ':'.join([path, streamname])
        os.remove(stream)

    @staticmethod
    def packData(path, data, streamname=None, verbose=False):

        files = ZlibUtil.compress(data, verbose)
        streamname = streamname or "packed_data"

        try:
           with tarfile.open(path + ":" + streamname, "w") as tar:
                for name in files:
                    fbuffer = io.BytesIO(files[name])
                    finfo = tarfile.TarInfo(name)
                    finfo.size = len(files[name])
                    tar.addfile(finfo, fileobj=fbuffer)
                    fbuffer.close()

        except IOError:
            pass

    @staticmethod
    def listData(path, streamname=None, verbose=False):
        streamname = streamname or "packed_data"
        with tarfile.open(path + ":" + streamname, "r") as tar:
            tar.list(verbose)

    @staticmethod
    def unpackData(filepath, outpath=None, streamname=None, verbose=False):
        streamname = streamname or "packed_data"
        path = outpath or os.getcwd()
        path = os.path.abspath(path) + os.path.sep
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))


        files = {}
        with tarfile.open(filepath + ":" + streamname, "r") as tar:
            x = tar.getnames()

            for item in x:
                file = tar.extractfile(item)
                files[item] = file.read()

        workdir = os.getcwd()
        os.chdir(path)

        ZlibUtil.decompressAll(files, verbose)

        os.chdir(workdir)

    @staticmethod
    def findStreams(path=None, show_special=False, recursive=False):
        path = path or os.getcwd()
        for file in glob.glob(path + '/*'):
            show_ads(file, show_special)
            if os.path.isdir(file) and recursive:
                NTFSDataStreams.findStreams(file, show_special, recursive)


    @staticmethod
    def listStreams(path=None, show_special=False):
        path = path or os.getcwd()
        show_ads(path, show_special)

    @staticmethod
    def catStream(path, streamname):
        if not os.path.exists: return

        with open(":".join([path, streamname]), 'r') as stream:
            print(stream.read())


class WIN32_FIND_STREAM_DATA(Structure):
    _fields_ = [("StreamSize", c_longlong), ("cStreamName", c_wchar*296)]


class WIN32_STREAM_ID(Structure):
    _pack_ = 1
    _fields_ = [("dwStreamId", c_long), ("dwStreamAttributes", c_long), ("Size", c_longlong), ("dwStreamNameSize", c_long)]


def get_ads(pathname):

    ads = []

    fsd = WIN32_FIND_STREAM_DATA()

    h = windll.kernel32.FindFirstStreamW(pathname, 0, byref(fsd), 0)

    if h != -1:
        while windll.kernel32.FindNextStreamW(h, byref(fsd)):
            if not fsd.cStreamName.endswith('$DATA'): continue
            se = StreamEntry(64*'\0')
            se.FileSize = fsd.StreamSize
            se.StreamName = fsd.cStreamName[1:-6].encode('utf-16le')
            se.wStreamNameLength = len(se.StreamName)
            se.SrcPathname = pathname+fsd.cStreamName[:-6]
            se.StreamId = 4
            ads += [se]
        windll.kernel32.CloseHandle(h)
    return ads

def get_ads_xp(pathname):

    hFile = windll.kernel32.CreateFileW(pathname, 0x80000000, 1, 0, 3, 0x02000000, 0)

    ads = []
    if hFile == -1: return ads

    buf = create_string_buffer(4096)
    context = c_int(0)

    while True:
        cb_read = c_int(0)

        if not windll.kernel32.BackupRead(hFile, buf, sizeof(WIN32_STREAM_ID), byref(cb_read), 0, 1, byref(context)):
            return ads

        if not cb_read.value: break

        wsid = cast(buf, POINTER(WIN32_STREAM_ID)).contents

        if wsid.dwStreamId == 0: break

        j = sizeof(WIN32_STREAM_ID)
        windll.kernel32.BackupRead(hFile, addressof(buf)+sizeof(WIN32_STREAM_ID), wsid.dwStreamNameSize, byref(cb_read), 0, 1, byref(context))
        s = cast(buf[j: j+wsid.dwStreamNameSize]+'\0\0'.encode('utf-16le'), c_wchar_p).value
        se = StreamEntry(64*'\0')
        se.FileSize = wsid.Size
        se.StreamName = s[1:-6]
        se.wStreamNameLength = len(se.StreamName)
        se.SrcPathname = pathname+s[:-6]
        se.StreamId = wsid.dwStreamId
        ads += [se]

        dwNextStreamLow, dwNextStreamHigh = c_int(0), c_int(0)
        windll.kernel32.BackupSeek(hFile, -1, -1, byref(dwNextStreamLow), byref(dwNextStreamHigh), byref(context))

    windll.kernel32.BackupRead(hFile, 0, 0, byref(cb_read), 1, 0, byref(context))
    windll.kernel32.CloseHandle(hFile)
    return ads

if sys.getwindowsversion().major < 6:
    get_ads = get_ads_xp

class StreamEntry:

    layout = {
        0x00: ('liLength', '<Q'),
        0x08: ('liUnused', '<Q'),
        0x10: ('bHash', '20s'),
        0x24: ('wStreamNameLength', '<H')
    }

    def __init__(self, s):
        self._i = 0
        self._pos = 0
        self._buf = s
        self._kv = StreamEntry.layout.copy()
        self._vk = {}
        for k, v in list(self._kv.items()):
            self._vk[v[0]] = k
        self.size = 0
        if self.wStreamNameLength:
            self.StreamName = (self._buf[0x26:0x26+self.wStreamNameLength]).decode('utf-16le')
        else:
            self.StreamName = ''

    def __getattr__(self, name):
        i = self._vk[name]
        fmt = self._kv[i][1]
        cnt = struct.unpack_from(fmt, self._buf.encode('utf-16le'), i+self._i) [0]
        setattr(self, name,  cnt)
        return cnt

def show_ads(file, show_special=False):
    dataId = {
        1: '$DATA',
        2: '$EA',
        3: '$SECURITY_DESCRIPTOR',
        4: '$DATA',
        5: '$FILE_NAME',
        6: '$PROPERTY_SET',
        7: '$OBJECT_ID',
        8: '$REPARSE_POINT',
        9: '$DATA'
    }

    stream_types={
        1: _("Standard data"),
        2: _("Extended attribute data"),
        3: _("Security descriptor data"),
        4: _("Alternative data stream"),
        5: _("Hard link information"),
        6: _("Property data"),
        7: _("Objects identifiers"),
        8: _("Reparse points"),
        9: _("Sparse file")
    }

    streams = get_ads(file)

    streams = list(filter(lambda x: x.StreamId == 4 or show_special, streams))

    if len(streams):
        print('\n' + os.path.abspath(file) + ':\n')

    for i in streams:
        print(' ' * 5, _('Path => {}').format(os.path.abspath(i.SrcPathname)))
        if i.StreamId == 4:  print(' ' * 5, _('Name: {}').format(i.StreamName))
        print(' ' * 5, _('Size: {}').format(sizeof_fmt(i.FileSize)))
        print(' ' * 5, _('Type: {}').format(dataId[i.StreamId]))
        print(' ' * 5, _('Description: {}\n').format(stream_types[i.StreamId]))
