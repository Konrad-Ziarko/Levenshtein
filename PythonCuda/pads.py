#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

_ = lambda s: s

__version__ = '1.0.1'

__doc__ = _("""pads -- Python NTFS Alternate Data Streams utility

pads is a utility that allows you to manage alternate data streams stored in the NTFS file system.

Usage: pads [options] inputfile

Options:

    -l path,  --list=path
        List alternate data streams attached to file or directory.
        Example: pads [-v] -l inputfile

    -d path,  --list-dir=path
        List alternate data streams attached to files in directory.
        Use -r flag for recursive mode.
        Example: pads [-r] [-v] -d [path]

    -x stream,  --extract=stream
        Extract alternate data stream from file or directory.
        Example: pads [-n filename] -x streamname inputfile

    -a path,  --add=path
        Add alternate data stream to file or directory.
        Example: pads [-n filename] -a streamfile inputfile

    -n name,  --file-name=name
        Provide new name for file or stream.

    -o path,  --output=path
        Provide output file name.

    -p path,  --pack=path
        Pack file or folder and add as stream.
        Example: pads [-v] [-n streamname] -p in_path inputfile

    -u,  --unpack
        Unpack and extract stream.
        Example: pads [-v] [-n streamname] [-o out_path] -u inputfile

    --list-packed
        List packed stream.
        Example: pads [-v] [-n streamname] --list-packed inputfile

    -r,  --recursive
        Recursive mode.

    -v,  --verbose
        Detailed mode.

    --remove=name
        Remove alternate data stream.
        Example: pads --remove=streamname inputfile

    -h,  --help
        Print this message and exit.

    -V,  --version
        Display version information and exit.
""")

from NTFSstreams import NTFSDataStreams
import os, sys, getopt
"""
def usage(code=0, msg=''):
    if msg:
        print(_("Error:"), msg, file=sys.stderr)
        print(_("\nTry 'pads --help' for more information."), file=sys.stderr)
    else:
        print(__doc__ % globals(), file=sys.stderr)

    sys.exit(code)
    """

def main():

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            'a:l:dx:n:o:p:urVvh',
            ['list=', 'list-dir', 'extract=', 'help',
             'file-name=', 'output=', 'add=', 'list-packed',
             'pack=', 'unpack', 'recursive', 'version', 'remove=',
             ])
    except getopt.error as msg:
        usage(1, msg)

    options = {
          'verbose' : False,
          'filename' : None,
          'output' : None,
          'recursive' : False,
        }


    for o, a in opts:
        if o in ("-l", "--list"):
            if os.path.isdir(a) or os.path.isfile(a):
                NTFSDataStreams.listStreams(a, show_special=options['verbose'])
            else:
                usage(1, _("No such file => {}.").format(a))

            sys.exit(0)

        elif o in ("-v", "--verbose"):
            options['verbose'] = True

        elif o in ("-d", "--list-dir"):
            if len(args) == 0:
                a = os.getcwd()
            elif len(args) == 1:
                a = args[0]
            else:
                usage(1, _('Too many arguments'))

            if os.path.isdir(a):
                NTFSDataStreams.findStreams(a, show_special=options['verbose'], recursive=options['recursive'])
            else:
                usage(1, _("{} is not a directory").format(a))

            sys.exit(0)

        elif o in ("-x", "--extract"):
            if len(args) != 1: usage(1, _('The program takes one argument'))
            if os.path.exists(args[0]) and os.path.exists(':'.join([args[0], a])):
                NTFSDataStreams.extract(args[0], a, options['filename'])
            else:
                usage(1, _("No such stream: {}.").format(':'.join([args[0], a])))

            sys.exit(0)

        elif o in ("-o", "--output"):
            options['output'] = a

        elif o in ("-n", "--file-name"):
            options['filename'] = a

        elif o in ("-a", "--add"):
            if len(args) != 1: usage(1, _('The program takes one argument'))
            if os.path.exists(args[0]):
                if os.path.exists(a):
                    NTFSDataStreams.add(args[0], a, options['filename'])
                else:
                    usage(1, _("No such file => {}").format(a))
            else:
                usage(1, _("No such file => {}").format(args[0]))

            sys.exit(0)

        elif o in ("-p", "--pack"):
            if len(args) != 1: usage(1, _('The program takes one argument'))
            if os.path.exists(args[0]):
                if os.path.exists(a):
                    NTFSDataStreams.packData(args[0], a, streamname=options['filename'], verbose=options['verbose'])
                else:
                    usage(1, _("No such file or directory: {}.").format(a))
            else:
                usage(1, _("No such file or directory: {}.").format(args[0]))

            sys.exit(0)

        elif o in ("-u", "--unpack"):
            if len(args) != 1: usage(1, _('The program takes one argument'))
            if os.path.exists(args[0]) and os.path.exists(':'.join([args[0], options['filename'] or "packed_data"])):
                NTFSDataStreams.unpackData(args[0], options['output'], streamname=options['filename'], verbose=options['verbose'])
            else:
                usage(1, _("No such stream: {}.").format(':'.join([args[0], options['filename'] or "packed_data"])))

            sys.exit(0)

        elif o == "--list-packed":
            if len(args) != 1: usage(1, _('The program takes one argument'))

            if os.path.exists(args[0]) and os.path.exists(':'.join([args[0], options['filename'] or "packed_data"])):
                NTFSDataStreams.listData(args[0], streamname=options['filename'], verbose=options['verbose'])
            else:
                usage(1, _("No such stream: {}.").format(':'.join([args[0], options['filename'] or "packed_data"])))

            sys.exit(0)

        elif o == "--remove":
            if len(args) != 1: usage(1, _('The program takes one argument'))

            if os.path.exists(args[0]) and os.path.exists(':'.join([args[0], a])): #remove
                NTFSDataStreams.remove(args[0], a)
            else:
                usage(1, _("No such stream: {}.").format(':'.join([args[0], a])))

            sys.exit(0)

        elif o in ("-r", "--recursive"):
            options['recursive'] = True

        elif o in ("-h", "--help"):
            usage(0)

        elif o in ("-V", "--version"):
            print(_('pads.py, %s') % __version__)

            sys.exit(0)

    if not args:
        usage(1, _('No input file given'))
    elif len(args) > 1:
        usage(1, _('Too many arguments'))

    if len(args) == 1:
        if os.path.isdir(args[0]) or os.path.isfile(args[0]):
            NTFSDataStreams.listStreams(args[0])
        else:
            usage(1, _("No such file => {}.").format(args[0]))

        sys.exit(0)

if __name__ == '__main__':
    main()
