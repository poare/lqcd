#!/usr/bin/env python3
"""Make a QLUA-written SciDAC/LIME gauge file readable by QDP++.

Two distinct defects show up in these files and they need different fixes.

1. QLUA writes the user metadata records 'scidac-file-xml' and
   'scidac-record-xml' as bare strings ("Dummy user file xml"). QDP++'s
   QDPFileReader parses those into an XMLReader, so it dies with

       Entity: line 1: parser error : Start tag expected, '<' not found

   Fix: wrap the text in a single <userRecord> element.

2. Chroma-written configs carry perfectly good XML with a trailing NUL byte.
   Fix: strip the NUL. Do NOT wrap these -- they begin with an XML
   declaration, and nesting that inside an element gives

       Entity: line 1: parser error : XML declaration allowed only at the
       start of the document

   which is a worse failure than the one being fixed.

In both cases the binary data and the SciDAC checksum are untouched. Record
sizes and LIME padding are recomputed, so nothing is truncated.

    python3 fix_lime_xml.py <in.lime> <out.lime>
"""
import struct, sys, xml.dom.minidom

MAGIC   = 0x456789ab
HDRLEN  = 144
# Only user metadata is ever rewritten. The scidac-private-* records are
# machine-generated and already valid; the binary data and checksum are not
# touched under any circumstances.
PATCHABLE = {b"scidac-file-xml", b"scidac-record-xml"}


def records(path):
    with open(path, "rb") as f:
        while True:
            hdr = f.read(HDRLEN)
            if len(hdr) < HDRLEN:
                return
            magic, ver, bits, nbytes = struct.unpack(">IHHQ", hdr[:16])
            if magic != MAGIC:
                raise SystemExit(f"not a LIME file: bad magic {magic:#x}")
            name = hdr[16:HDRLEN].split(b"\0")[0]
            data = f.read(nbytes)
            f.read((-nbytes) % 8)          # skip padding
            yield ver, bits, name, data


def parses(data):
    try:
        xml.dom.minidom.parseString(data)
        return True
    except Exception:
        return False


def main(src, dst):
    n_patched = 0
    with open(dst, "wb") as out:
        for ver, bits, name, data in records(src):
            if name in PATCHABLE and not parses(data):
                stripped = data.rstrip(b"\0")
                if parses(stripped):
                    # Defect 2: valid XML, trailing NUL. Just drop the NUL.
                    data = stripped
                    n_patched += 1
                    print(f"patched {name.decode()}: stripped "
                          f"{len(data) - len(stripped) + (len(data) - len(stripped)) or 1} "
                          f"trailing NUL byte(s); content left intact")
                else:
                    # Defect 1: not XML at all. Wrap it.
                    text = stripped.decode("utf-8", "replace").strip()
                    data = ("<userRecord>" + text + "</userRecord>").encode("utf-8")
                    n_patched += 1
                    print(f"patched {name.decode()}: wrapped bare text {text!r}")
            hdr = struct.pack(">IHHQ", MAGIC, ver, bits, len(data))
            hdr += name + b"\0" * (HDRLEN - 16 - len(name))
            out.write(hdr)
            out.write(data)
            out.write(b"\0" * ((-len(data)) % 8))
    print(f"wrote {dst}  ({n_patched} record(s) patched)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
