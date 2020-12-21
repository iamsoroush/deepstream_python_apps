import sys
sys.path.append('../')
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
from time import sleep

import numpy as np
import cv2

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import GObject, Gst, GstVideo
from common.bus_call import bus_call
from common.FPS import GETFPS

import pyds

from gstutils import get_num_channels, get_np_dtype


PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3


def gst_to_np(sample):
    buffer = sample.get_buffer()
    # print('buffer: ', buf)
    # print(f'pts: {buffer.pts / 1e9} -- dts: {buffer.dts / 1e9} -- offset: {buffer.offset} -- duration: {buffer.duration / 1e9}')
    caps = sample.get_caps()
    print(caps.get_structure(0).get_value('format'))
    print(caps.get_structure(0).get_value('height'))
    print(caps.get_structure(0).get_value('width'))
    # print('n_meta: ', buffer.get_n_meta())

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    l_frame = batch_meta.frame_meta_list
    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
    frame_number = frame_meta.frame_num
    pts = frame_meta.buf_pts
    ntp_ts = frame_meta.ntp_timestamp
    print(f'frame number: {frame_number}')
    print(f'frame pts (seconds): {pts / 1e9}')
    print(f'ntp timestamp (seconds): {ntp_ts / 1e9}')

    caps_format = caps.get_structure(0)
    video_format = GstVideo.VideoFormat.from_string(
        caps_format.get_value('format'))

    w, h = caps_format.get_value('width'), caps_format.get_value('height')
    c = get_num_channels(video_format)

    buffer_size = buffer.get_size()
    shape = (h, w, c) if (h * w * c == buffer_size) else buffer_size
    array = np.ndarray(shape=shape, buffer=buffer.extract_dup(0, buffer_size),
                       dtype=get_np_dtype(video_format))

    return np.squeeze(array), buffer.pts


def new_buffer(sink, data):
    sample = sink.emit("pull-sample")

    arr, pts = gst_to_np(sample)
    cv2.imwrite(f'{pts}.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    sleep(2)
    return Gst.FlowReturn.OK


class Pipeline:

    def __init__(self,
                 input_file_path,
                 model_config_path='./model/config_infer_primary_detectnet_v2.txt',
                 labels_path='./model/detectnet_v2_labels.txt',
                 output_file_path='./out.mp4'):
        self.model_config_path = model_config_path
        self.labels_path = labels_path
        self.output_file_path = output_file_path

        self.width = 1280
        self.height = 720

        GObject.threads_init()
        Gst.init(None)

        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        self.source, self.h264parser, self.decoder = self._create_source_elements(input_file_path)
        self.streammux, self.pgie = self._create_middle_elements()
        self.nvvidconv, self.capsfilter, self.sink = self._create_sink_elements()

        # Link the elements
        print("Linking elements in the Pipeline \n")
        self._link()

        # osdsinkpad = self.nvosd.get_static_pad("sink")
        # if not osdsinkpad:
        #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
        #
        # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        self.loop = GObject.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::eos", self._bus_call, self.loop)

    def start(self):
        # start play back and listen to events
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop.run()

    def _create_source_elements(self, file_path):
        # Source element for reading from the file
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        # Since the data format in the input file is elementary h264 stream,
        # we need a h264parser
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")

        # Use nvdec_h264 for hardware accelerated decode on GPU
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

        source.set_property('location', file_path)

        self.pipeline.add(source)
        self.pipeline.add(h264parser)
        self.pipeline.add(decoder)
        return source, h264parser, decoder

    def _create_middle_elements(self):
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        # # Use convertor to convert from NV12 to RGBA as required by nvosd
        # nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        # if not nvvidconv:
        #     sys.stderr.write(" Unable to create nvvidconv \n")

        # Create OSD to draw on the converted RGBA buffer
        # nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        # if not nvosd:
        #     sys.stderr.write(" Unable to create nvosd \n")
        #
        # nvosd.set_property('display-clock', 1)  # here: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsosd.html

        streammux.set_property('width', self.width)
        streammux.set_property('height', self.height)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)
        pgie.set_property('config-file-path', self.model_config_path)

        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        # self.pipeline.add(nvvidconv)
        # self.pipeline.add(nvosd)

        return streammux, pgie

    def _create_sink_elements(self):
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor appsink")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv2 \n")

        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if not capsfilter:
            sys.stderr.write(" Unable to create capsfilter \n")

        caps = Gst.Caps.from_string("video/x-raw, format=RGBA")
        capsfilter.set_property("caps", caps)

        sink = Gst.ElementFactory.make("appsink", "sink")
        if not sink:
            sys.stderr.write(" Unable to create appsink \n")
        sink.set_property("emit-signals", True)
        caps = Gst.caps_from_string("video/x-raw, format=RGBA")
        sink.set_property("caps", caps)
        sink.set_property("drop", True)
        sink.set_property("max_buffers", 3)
        # sink.set_property("sync", False)
        sink.set_property("wait-on-eos", False)
        sink.connect("new-sample", new_buffer, sink)



        self.pipeline.add(nvvidconv)
        self.pipeline.add(capsfilter)
        self.pipeline.add(sink)

        return nvvidconv, capsfilter, sink

    def _link(self):
        self.source.link(self.h264parser)
        self.h264parser.link(self.decoder)

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")
        srcpad = self.decoder.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")

        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.nvvidconv)
        self.nvvidconv.link(self.capsfilter)
        self.capsfilter.link(self.sink)

    @staticmethod
    def _bus_call(bus, message, loop):
        print('buss called on {}'.format(message))
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        return True

    @staticmethod
    def osd_sink_pad_buffer_probe(pad, info, u_data):
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                # frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    # obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                obj_counter[obj_meta.class_id] += 1
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.

            fps_stream.get_fps()

            py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}" \
                .format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # if WRITE_FRAMES:
            #     n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            #     # convert python array into numy array format.
            #     frame_image = np.array(n_frame, copy=True, order='C')
            #     # covert the array into cv2 default color format
            #     frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGRA)
            #     cv2.imwrite("./frame_" + str(frame_number) + ".jpg",
            #                 frame_image)
                # print('saved to')

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


class PipelineCamera:

    def __init__(self,
                 model_config_path='./model/config_infer_primary_detectnet_v2.txt',
                 labels_path='./model/detectnet_v2_labels.txt',
                 output_file_path='./out.mp4'):
        self.model_config_path = model_config_path
        self.labels_path = labels_path
        self.output_file_path = output_file_path

        self.width = 1280
        self.height = 720

        GObject.threads_init()
        Gst.init(None)

        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        self.source, self.nvvidconv_src, self.caps_nvvidconv_src = self._create_source_elements()
        self.streammux, self.pgie = self._create_middle_elements()
        self.nvvidconv, self.capsfilter, self.sink = self._create_sink_elements()

        # Link the elements
        print("Linking elements in the Pipeline \n")
        self._link()

        # osdsinkpad = self.nvosd.get_static_pad("sink")
        # if not osdsinkpad:
        #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
        #
        # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        self.loop = GObject.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::eos", self._bus_call, self.loop)

    def start(self):
        # start play back and listen to events
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop.run()

    def _create_source_elements(self):
        source = Gst.ElementFactory.make("nvarguscamerasrc", "src-elem")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        # Converter to scale the image
        nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
        if not nvvidconv_src:
            sys.stderr.write(" Unable to create nvvidconv_src \n")

        # Caps for NVMM and resolution scaling
        caps_nvvidconv_src = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        if not caps_nvvidconv_src:
            sys.stderr.write(" Unable to create capsfilter \n")

        source.set_property('bufapi-version', True)
        caps_nvvidconv_src.set_property('caps', Gst.Caps.from_string(
            'video/x-raw(memory:NVMM), width={}, height={}'.format(self.width, self.height)))

        self.pipeline.add(source)
        self.pipeline.add(nvvidconv_src)
        self.pipeline.add(caps_nvvidconv_src)
        return source, nvvidconv_src, caps_nvvidconv_src

    def _create_middle_elements(self):
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        # Use nvinfer to run inferencing on decoder's output,
        # behaviour of inferencing is set through config file
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        # # Use convertor to convert from NV12 to RGBA as required by nvosd
        # nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        # if not nvvidconv:
        #     sys.stderr.write(" Unable to create nvvidconv \n")

        # Create OSD to draw on the converted RGBA buffer
        # nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        # if not nvosd:
        #     sys.stderr.write(" Unable to create nvosd \n")
        #
        # nvosd.set_property('display-clock', 1)  # here: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsosd.html

        streammux.set_property('width', self.width)
        streammux.set_property('height', self.height)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)
        pgie.set_property('config-file-path', self.model_config_path)

        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        # self.pipeline.add(nvvidconv)
        # self.pipeline.add(nvosd)

        return streammux, pgie

    def _create_sink_elements(self):
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor appsink")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv2 \n")

        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if not capsfilter:
            sys.stderr.write(" Unable to create capsfilter \n")

        caps = Gst.Caps.from_string("video/x-raw, format=RGBA")
        capsfilter.set_property("caps", caps)

        sink = Gst.ElementFactory.make("appsink", "sink")
        if not sink:
            sys.stderr.write(" Unable to create appsink \n")
        sink.set_property("emit-signals", True)
        caps = Gst.caps_from_string("video/x-raw, format=RGBA")
        sink.set_property("caps", caps)
        sink.set_property("drop", True)
        sink.set_property("max_buffers", 3)
        # sink.set_property("sync", False)
        sink.set_property("wait-on-eos", False)
        sink.connect("new-sample", new_buffer, sink)

        self.pipeline.add(nvvidconv)
        self.pipeline.add(capsfilter)
        self.pipeline.add(sink)

        return nvvidconv, capsfilter, sink

    def _link(self):
        self.source.link(self.nvvidconv_src)
        self.nvvidconv_src.link(self.caps_nvvidconv_src)

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")
        srcpad = self.caps_nvvidconv_src.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")

        srcpad.link(sinkpad)
        self.streammux.link(self.pgie)
        self.pgie.link(self.nvvidconv)
        self.nvvidconv.link(self.capsfilter)
        self.capsfilter.link(self.sink)

    @staticmethod
    def _bus_call(bus, message, loop):
        print('buss called on {}'.format(message))
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        return True

    @staticmethod
    def osd_sink_pad_buffer_probe(pad, info, u_data):
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                # frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    # obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                obj_counter[obj_meta.class_id] += 1
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.

            fps_stream.get_fps()

            py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}" \
                .format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # if WRITE_FRAMES:
            #     n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            #     # convert python array into numy array format.
            #     frame_image = np.array(n_frame, copy=True, order='C')
            #     # covert the array into cv2 default color format
            #     frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGRA)
            #     cv2.imwrite("./frame_" + str(frame_number) + ".jpg",
            #                 frame_image)
                # print('saved to')

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


if __name__ == '__main__':
    fps_stream = GETFPS(0)

    out_file_name = '{}.mp4'.format(sys.argv[1])
    in_file_path = sys.argv[2]

    # pipeline = Pipeline(output_file_path=out_file_name)
    # pipeline = Pipeline(in_file_path, output_file_path=out_file_name)
    pipeline = PipelineCamera(output_file_path=out_file_name)
    try:
        pipeline.start()
    except KeyboardInterrupt as e:
        # sink.get_static_pad('sink').send_event(Gst.Event.new_eos())
        # pipeline.send_event(Gst.Event.new_eos())
        # pipeline.set_state(Gst.State.NULL)
        pipeline.pipeline.send_event(Gst.Event.new_eos())

        # Wait for EOS to be catched up by the bus
        msg = pipeline.bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.EOS
        )
        print(msg)
        sleep(5)
    except Exception as e:
        print(e)
    finally:
        pipeline.pipeline.set_state(Gst.State.NULL)