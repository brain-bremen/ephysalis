"""Reference values from the MCS Neuroshare library (32-bit nsMCDLibrary.dll, ctypes), for
checking an MCD reader (docs/legacy/mcd.md). The DLL is 32-bit, so run this with a 32-bit
Python on Windows (e.g. ``uv python install cpython-3.11-windows-x86-none``):

    python mcd_neuroshare_reference.py <nsMCDLibrary.dll> <file.mcd> <out.json> [max samples per entity]

The library is part of MCS's Neuroshare download; it is not distributed with ephysalis. It can mix
up files with the same name in different folders (it returned another folder's Data.mcd).

Writes entity infos, analog infos, the first samples of every analog entity (as doubles, in
the entity's units) and all events (timestamps, raw bytes as hex)."""

import ctypes as C
import json
import sys


class FileInfo(C.Structure):
    _pack_ = 4
    _fields_ = [("szFileType", C.c_char * 32), ("dwEntityCount", C.c_uint32), ("dTimeStampResolution", C.c_double),
                ("dTimeSpan", C.c_double), ("szAppName", C.c_char * 64), ("dwTime_Year", C.c_uint32),
                ("dwTime_Month", C.c_uint32), ("dwTime_DayofWeek", C.c_uint32), ("dwTime_Day", C.c_uint32),
                ("dwTime_Hour", C.c_uint32), ("dwTime_Min", C.c_uint32), ("dwTime_Sec", C.c_uint32),
                ("dwTime_MilliSec", C.c_uint32), ("szFileComment", C.c_char * 256)]


class EntityInfo(C.Structure):
    _pack_ = 4
    _fields_ = [("szEntityLabel", C.c_char * 32), ("dwEntityType", C.c_uint32), ("dwItemCount", C.c_uint32)]


class EventInfo(C.Structure):
    _pack_ = 4
    _fields_ = [("dwEventType", C.c_uint32), ("dwMinDataLength", C.c_uint32), ("dwMaxDataLength", C.c_uint32),
                ("szCSVDesc", C.c_char * 128)]


class AnalogInfo(C.Structure):
    _pack_ = 4
    _fields_ = [("dSampleRate", C.c_double), ("dMinVal", C.c_double), ("dMaxVal", C.c_double),
                ("szUnits", C.c_char * 16), ("dResolution", C.c_double), ("dLocationX", C.c_double),
                ("dLocationY", C.c_double), ("dLocationZ", C.c_double), ("dLocationUser", C.c_double),
                ("dHighFreqCorner", C.c_double), ("dwHighFreqOrder", C.c_uint32), ("szHighFilterType", C.c_char * 16),
                ("dLowFreqCorner", C.c_double), ("dwLowFreqOrder", C.c_uint32), ("szLowFilterType", C.c_char * 16),
                ("szProbeInfo", C.c_char * 128)]


class SegmentInfo(C.Structure):
    _pack_ = 4
    _fields_ = [("dwSourceCount", C.c_uint32), ("dwMinSampleCount", C.c_uint32), ("dwMaxSampleCount", C.c_uint32),
                ("dSampleRate", C.c_double), ("szUnits", C.c_char * 32)]


def as_dict(s):
    return {k: (getattr(s, k).decode("latin-1") if isinstance(getattr(s, k), bytes) else getattr(s, k))
            for k, _ in s._fields_}


def main():
    dll_path, mcd, out = sys.argv[1:4]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    lib = C.CDLL(dll_path)
    handle = C.c_uint32()
    assert lib.ns_OpenFile(mcd.encode(), C.byref(handle)) == 0
    info = FileInfo()
    assert lib.ns_GetFileInfo(handle, C.byref(info), C.sizeof(info)) == 0
    result = {"file": as_dict(info), "entities": []}
    for e in range(info.dwEntityCount):
        ent = EntityInfo()
        lib.ns_GetEntityInfo(handle, e, C.byref(ent), C.sizeof(ent))
        item = {"entity": as_dict(ent)}
        if ent.dwEntityType == 2:  # analog
            a = AnalogInfo()
            lib.ns_GetAnalogInfo(handle, e, C.byref(a), C.sizeof(a))
            item["analog"] = as_dict(a)
            n = min(ent.dwItemCount, max_samples)
            data = (C.c_double * max(n, 1))()
            count = C.c_uint32()
            item["status"] = lib.ns_GetAnalogData(handle, e, 0, n, C.byref(count), data)
            item["cont_count"] = count.value
            item["samples"] = list(data[:n])
            t = C.c_double()
            lib.ns_GetTimeByIndex(handle, e, 0, C.byref(t))
            item["t0"] = t.value
        elif ent.dwEntityType == 1:  # event
            ev = EventInfo()
            lib.ns_GetEventInfo(handle, e, C.byref(ev), C.sizeof(ev))
            item["event"] = as_dict(ev)
            events = []
            for k in range(min(ent.dwItemCount, max_samples)):
                t = C.c_double()
                buf = (C.c_ubyte * 256)()
                ret = C.c_uint32()
                lib.ns_GetEventData(handle, e, k, C.byref(t), buf, 256, C.byref(ret))
                events.append([t.value, bytes(buf[:ret.value]).hex()])
            item["events"] = events
        elif ent.dwEntityType == 3:  # segment
            s = SegmentInfo()
            lib.ns_GetSegmentInfo(handle, e, C.byref(s), C.sizeof(s))
            item["segment"] = as_dict(s)
        result["entities"].append(item)
    lib.ns_CloseFile(handle)
    with open(out, "w") as f:
        json.dump(result, f)


main()
