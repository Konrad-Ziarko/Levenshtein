# -*- coding: UTF-8 -*-

import zlib, glob, os

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)

class ZlibUtil:

    @staticmethod
    def compress(path, verbose=False):
        files = {}

        if os.path.isdir(path):
            for file in glob.glob(path + "/*"):

                files.update(ZlibUtil.compress(file, verbose))

        elif os.path.isfile(path):

            try:
                with open(path, 'rb') as data:
                    indata = data.read()
                    outdata = zlib.compress(indata, zlib.Z_BEST_COMPRESSION)
                    files[path] = outdata

                    if verbose: print(path, sizeof_fmt(len(indata)), "=>", sizeof_fmt(len(outdata)), "%d%%" % (len(outdata) * 100 / len(indata)))

            except IOError:
                pass
        else:
            raise OSError(2, 'No such file or directory', path)
        return files

    @staticmethod
    def decompress(filename, filedata, verbose=False):
            outdata = zlib.decompress(filedata)

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            with open(filename, 'wb') as data:
                data.write(outdata)

                if verbose: print(
                    filename,
                    sizeof_fmt(len(filedata)),
                    "=>",
                    os.getcwd() + os.path.sep + filename,
                    sizeof_fmt(len(outdata)),
                    "%d%%" % (len(outdata) * 100 / len(filedata))
                )

    @staticmethod
    def decompressAll(files, verbose=False):
        for file in files:
            ZlibUtil.decompress(file, files[file], verbose)


