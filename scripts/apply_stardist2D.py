from omero.util.tiles import TileLoopIteration, RPSTileLoop
from omero.model import PixelsI
from omero.gateway import BlitzGateway, MapAnnotationWrapper, TagAnnotationWrapper
from omero.rtypes import rlong, rstring, robject
from omero.constants import metadata
import omero.scripts as scripts
import ezomero
import omero
import omero_rois
import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist.big import BlockND
from stardist.matching import relabel_sequential


def add_map_annotation(conn, key_value_data, image_id):
    '''Add map annotation to image'''
    map_ann = MapAnnotationWrapper(conn)
    # Use 'client' namespace to allow editing in Insight & web
    namespace = metadata.NSCLIENTMAPANNOTATION
    map_ann.setNs(namespace)
    map_ann.setValue(key_value_data)
    map_ann.save()
    omero_image = conn.getObject("Image", image_id)
    omero_image.linkAnnotation(map_ann)
    return


def add_tag_annotation(conn, tag_text, image_id):
    '''Add a tag to an image'''
    # tag_ann = None
    # for tag in conn.getObjects("TagAnnotation"):
    #     if tag_text == tag.textValue:
    #         tag_ann = tag
    # if not tag_ann:
    tag_ann = TagAnnotationWrapper(conn)
    tag_ann.setValue(tag_text)
    tag_ann.save()

    omero_image = conn.getObject("Image", image_id)
    omero_image.linkAnnotation(tag_ann)
    # clean_duplicated_tags(conn)

    return


def clean_duplicated_tags(conn):
    # create a list of tags and append only the tags, tag id and tag owner id to it
    tags = []

    for tag in conn.getObjects("TagAnnotation"):
        owner = tag.getDetails().owner.id.val
        # print tag.textValue, owner
        tags.append([tag.textValue, str(tag.id), str(owner)])

    # sort the tags in descending order to allow to see duplicates

    tags.sort(key=lambda tag: tag[0].lower())
    print("sorted tags", tags)

    prev_tag = ""
    prev_id = 0
    for t in tags:
        tag_id = str(t[1])
        if t[0] == prev_tag:
            # move all tagged objects to previous tags and delete
            for link in conn.getAnnotationLinks('Image', ann_ids=[tag_id]):
                link._obj.child = omero.model.TagAnnotationI(prev_id, False)
                link.save()
            conn.deleteObjects('TagAnnotation', [tag_id])
        prev_tag = t[0]
        prev_id = tag_id
    return

# We have a helper function for creating an ROI and linking it to new shapes


def create_roi(updateService, img, shapes):
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)


def make_rgba_list(label_image):
    from skimage import color
    import numpy as np
    unique_labels = (color.label2rgb(np.unique(label_image))*255).astype(int)
    rgba_list = []
    for row in range(unique_labels.shape[0]):
        rgba_list.append(tuple(unique_labels[row].tolist() + [128]))
    return rgba_list

# Another helper for generating the color integers for shapes


def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r+g+b+a
    if (rgba_int > (2**31-1)):       # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32
    return rgba_int


def create_image_from_tiles(conn, source, blocks, labels, image_name, model_name, description, tile_size):
    '''
    source: image_object'
    box: (x, y, w, h, z1, z2, t1, t2, xy_by_time)
    tile_size: block_size (int)
    '''

    pixels_service = conn.getPixelsService()
    query_service = conn.getQueryService()
    # xbox, ybox, wbox, hbox, z1box, z2box, t1box, t2box, xy_by_time = box
    # wbox, hbox = box
    size_x = labels.shape[1]
    size_y = labels.shape[0]
    size_z = source.getSizeZ()
    size_t = source.getSizeT()
    size_c = source.getSizeC()
    tile_width = tile_size
    tile_height = tile_size
    # primary_pixels = source.getPrimaryPixels()

    # Creates Omero image from numpy array
    label_image_name = image_name + "_label_" + model_name

    def create_image():
        query = "from PixelsType as p where p.value='uint32'"
        pixels_type = query_service.findByQuery(query, None)
        print("pixels_type", pixels_type)
        # pixels_type = labels.dtype.name
        channel_list = range(size_c)
        # bytesPerPixel = pixelsType.bitSize.val / 8
        iid = pixels_service.createImage(
            size_x,
            size_y,
            size_z,
            size_t,
            channel_list,
            pixels_type,
            image_name,
            description,
            conn.SERVICE_OPTS)

        return conn.getObject("Image", iid)

    def get_tiles(blocks):
        for block in blocks:
            slc = block.slice_read()
            y_start, y_stop = slc[0].start, slc[0].stop
            x_start, x_stop = slc[1].start, slc[1].stop
            tile = labels[y_start:y_stop, x_start:x_stop]
            # tile = tile[...,np.newaxis,np.newaxis,np.newaxis]
        #     tiles.append(tile)
        # for tile in tiles:
            yield tile
    tile_gen = get_tiles(blocks)

    def next_tile():
        return next(tile_gen)

    class Iteration(TileLoopIteration):

        def run(self, data, z, c, t, x, y, tile_width, tile_height,
                tile_count):
            current_tile2d = next_tile()
            print("current_tile2d type", type(current_tile2d))
            print("current_tile2d shape", current_tile2d.shape)
            print(current_tile2d[0, 0])
            print(current_tile2d.dtype)
            data.setTile(current_tile2d, z, c, t, x,
                         y, tile_width, tile_height)

    new_image = create_image()
    pid = new_image.getPixelsId()
    loop = RPSTileLoop(conn.c.sf, PixelsI(pid, False))
    loop.forEachTile(tile_width, tile_height, Iteration())

    for the_c in range(size_c):
        pixels_service.setChannelGlobalMinMax(pid, the_c, float(0),
                                              float(255), conn.SERVICE_OPTS)

    return new_image


def apply_stardist2D(conn, scriptParams):
    '''Apply a stardist 2D model to image(s) and saves label_image(s)'''
    # Initialize service
    updateService = conn.getUpdateService()
    # Get parameters
    dataType = scriptParams["Data_Type"]
    ids = scriptParams["IDs"]
    chosen_model = scriptParams["Available_Pretrained_Models"]
    multichannel_flag = scriptParams["Multichannel"]
    if multichannel_flag == True:
        ch = int(scriptParams["Channel_Number"])
    else:
        ch = 0
    print('channel = ', ch, type(ch))

    # Get list of images
    image_list, name_list = [], []
    if dataType == 'Dataset':
        dataset_id = ids[0]
        dataset = conn.getObject(dataType, dataset_id)
        image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)
        # for ds in objects:
        #     print("Processing Images from Dataset: %s" % ds.getName())
        #     imgs = list(ds.listChildren())
        #     image_list.extend(imgs)
        #     name_list = [img.getName() for img in imgs]
    else:
        print("Processing Images identified by ID")
        image_ids = ids
        # Get objects
        # generator of images or datasets
        obs = conn.getObjects(dataType, ids)
        objects = list(obs)
        # get first image to access dataset id
        im_object = objects[0]
        # get dataset
        dataset = im_object.getParent()
        print("From Dataset: ", dataset.getName())
        dataset_id = dataset.getId()

        # image_list = objects
        # name_list.append(image_list[0].getName())
        # # get dataset
        # dataset = image_list[0].getParent()
        # print("From Dataset: ", dataset.getName())
        # dataset_id = dataset.getId()

    # Stardist model
    # creates a pretrained model
    model = StarDist2D.from_pretrained(chosen_model)

    # Process images in python
    for image_id in image_ids:

        # Get only OMERO image object
        im_object, _ = ezomero.get_image(conn, image_id, no_pixels=True)
        name = im_object.getName()
        # Get image shape information
        image_shape = (im_object.getSizeX(),
                       im_object.getSizeY(),
                       im_object.getSizeZ(),
                       im_object.getSizeC(),
                       im_object.getSizeT())
        print('image_shape = ', image_shape)
        # Get pyramid levels
        try:
            levels = ezomero.get_pyramid_levels(conn, image_id)
        except AttributeError:
            levels = []
        print('levels = ', levels)
        # If image is in pyramid structure, apply stardist to tiles of highest resolution
        if (levels) and (len(levels) > 1):
            # Set up some stardist block arguments
            shape = image_shape[:2]
            block_size = min(levels[-3])
            min_overlap = int(block_size//10)
            context = min_overlap
            grid = 1
            print('block_size = ', block_size)
            axes = 'YX'
            # create block cover
            blocks = BlockND.cover(
                shape, axes, block_size, min_overlap, context=min_overlap, grid=1)
            # Create empty output for labels
            labels = np.zeros(shape, dtype=np.uint32)

            label_offset = 1
            for block in blocks:
                # Get tile from block_slices (in OMERO, these slices should be used to ezomero.get_image's start_coords and axis_lengths arguments )
                slc = block.slice_read()
                y_start, y_stop = slc[0].start, slc[0].stop
                x_start, x_stop = slc[1].start, slc[1].stop
                print('\nstart_coords = ', y_start, x_start)
                print('\nstop_coords = ', y_stop, x_stop)
                _, tile = ezomero.get_image(conn,
                                            image_id,
                                            start_coords=(
                                                y_start, x_start, 0, ch, 0),
                                            axis_lengths=(
                                                y_stop - y_start, x_stop - x_start, 1, 1, 1),
                                            dim_order='yxzct'
                                            )
                print(tile.shape)
                tile2d = tile.squeeze()
                print(tile2d.shape)
                ########
                # Here maybe check if tile has lots of zeros and circunvent predictions for that tile
                # It may also be useful to provide level number and min_overlap percentage to advanced part of user interface
                ########
                # Perform local predictions to normalized tile
                labels_local, polys = model.predict_instances(
                    normalize(tile2d))
                # Filter objects (match edges)
                labels_local = block.crop_context(labels_local)
                labels_local, polys = block.filter_objects(labels_local, polys)
                # Relabel labels sequentially from last offset
                labels_local = relabel_sequential(
                    labels_local, label_offset)[0]
                # Write processed block (labels) to
                block.write(labels, labels_local)
                # Update labels offset
                label_offset += len(polys['prob'])

            # tiles = []
            # def tile_gen():
            #     for block in blocks:
            #         slc = block.slice_read()
            #         y_start, y_stop = slc[0].start, slc[0].stop
            #         x_start, x_stop = slc[1].start, slc[1].stop
            #         tile = labels[y_start:y_stop, x_start:x_stop]
            #         # tile = tile[...,np.newaxis,np.newaxis,np.newaxis]
            #         tiles.append(tile)
            #     for tile in tiles:
            #         yield tile
            # # Creates Omero image from numpy array
            # label_image_name = name + "_label_" + chosen_model
            # desc = "labeled image"
            # omero_image = conn.createImageFromNumpySeq(
            #     tile_gen(), label_image_name, 1, 1, 1, description=desc,
            #     dataset=dataset)
            desc = "labeled image"
            tile_size = block_size
            new_tiled_image = create_image_from_tiles(conn, im_object, blocks, labels, name, chosen_model, desc,
                                                      tile_size)
            print('new_tiled_image = ', new_tiled_image)
        # If image is not in pyramid structure, apply stardist to whole image
        else:
            _, im = ezomero.get_image(conn, image_id,
                                      xyzct=True)
            im = im[:, :, 0, ch, 0]  # 2D image
            # Applies stardist model
            labels, _ = model.predict_instances(normalize(im))
            # Creates Omero image from numpy array
            planes = [labels]

            def plane_gen():
                """generator will yield planes"""
                for p in planes:
                    yield p
            label_image_name = name + "_label_" + chosen_model
            desc = "labeled image"
            omero_image = conn.createImageFromNumpySeq(
                plane_gen(), label_image_name, 1, 1, 1, description=desc,
                dataset=dataset)
        print('labels_shape = ', labels.shape)

        # Create ROIs from label image
        msks = omero_rois.masks_from_label_image(
            labels, raise_on_no_mask=False)
        rgba_list = make_rgba_list(labels)
        for msk, rgba in zip(msks, rgba_list[1:]):
            msk.fillColor = omero.rtypes.rint(rgba_to_int(*rgba))
        create_roi(updateService, im_object, msks)

        # Reshape to match omero standards
        # labels = labels[:,:,np.newaxis, np.newaxis, np.newaxis] # make it xyzct
        # save label image in the same dataset

        # im_id = ezomero.post_image(conn, labels, label_image_name,
        #                            dataset_id=dataset_id,
        #                            dim_order = 'yxzct') # xyzct led to rotated

        print('Created new Image:%s Name:"%s"' %
              (omero_image.getId(), omero_image.getName()))
        im_id = omero_image.getId()

        # Add key_values indicating source image and stardist model used
        key_value_data = [["Source_Image", name],
                          ["Stardist_Model", chosen_model],
                          ["Channel_Number", str(ch)],
                          ["Number of Nuclei", str(np.amax(labels))]]
        add_map_annotation(conn, key_value_data, im_id)

        # Add tags indicating label image from stardist2D
        tags = ["label", "stardist2D"]
        [add_tag_annotation(conn, tag, im_id) for tag in tags]

    return


if __name__ == "__main__":
    """
    The main entry point of the script, as called by the client via the
    scripting service, passing the required parameters.
    """

    dataTypes = [rstring('Dataset'), rstring('Image')]

    available_2D_models = [rstring('2D_versatile_fluo'),
                           rstring('2D_versatile_he'),
                           rstring('2D_paper_dsb2018')]

    client = scripts.client(
        'Apply_stardist2D.py',

        ("Label images with Stardist"),

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="The data you want to work with.", values=dataTypes,
            default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Dataset IDs or Image IDs").ofType(rlong(0)),

        scripts.String(
            "Available_Pretrained_Models", optional=False, grouping="3",
            description="List of available pretrained Stardist models",
            values=available_2D_models, default="2D_versatile_fluo"),

        # If multichannel, ask which channel contains nuclei
        scripts.Bool("Multichannel", grouping="4", default=False),

        scripts.String(
            "Channel_Number", optional=True, grouping="4.1",
            description="The channel number (first channel is 0).")

        # TO DO: allow using custom stardist2D model from file
        # scripts.Bool("Use_Custom_Model", grouping="4", default=False),

        # scripts.String(
        #     "Model_Name", optional=True, grouping="4.1",
        #     description="The model file name.")

    )

    try:
        # Get parameters
        scriptParams = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                # unwrap rtypes to String, Integer etc
                scriptParams[key] = client.getInput(key, unwrap=True)

        # wrap client to use the Blitz Gateway
        conn = BlitzGateway(client_obj=client)

        # process images
        processed_images = apply_stardist2D(conn, scriptParams)

    finally:
        client.closeSession()
