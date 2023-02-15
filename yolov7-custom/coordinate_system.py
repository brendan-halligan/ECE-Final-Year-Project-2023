def bilinear_interpolation(x, y, img_corners):
    """
    Function which performs bilinear interpolation to determine the Degree Decimal coordinates for a given pixel point.

    :param x:           Value of the pixel coordinate's X location.
    :param y:           Value of the pixel coordinate's Y location.
    :param img_corners: An array which contains four (latitude, longitude) tuples in Degrees Minutes Seconds (DMS) form.
                        Each tuple corresponds to a corner in the image. The order of corner tuples is as follows:
                            1. Top Left.
                            2. Top Right.
                            3. Bottom Right.
                            4. Bottom Left.
    :return:            A tuple containing the latitude and longitude coordinates of a particular pixel point in Degree
                        Decimal (DD) form, correct to the fourth decimal point.
    """
    # Process the coordinates provided for each corner.
    x1, y1, lat1, lon1 = img_corners[0]
    x2, y2, lat2, lon2 = img_corners[1]
    x3, y3, lat3, lon3 = img_corners[2]
    x4, y4, lat4, lon4 = img_corners[3]

    """
    Calculate the longitude and latitude values for the provided pixel point. 
    NB: The formulae utilised below correspond to those shown on the following webpage:
        https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    f12 = (y - y1) / (y2 - y1) * (lon2 - lon1) + lon1
    f34 = (y - y3) / (y4 - y3) * (lon4 - lon3) + lon3
    longitude = (x - x1) / (x4 - x1) * (f34 - f12) + f12

    f12 = (y - y1) / (y2 - y1) * (lat2 - lat1) + lat1
    f34 = (y - y3) / (y4 - y3) * (lat4 - lat3) + lat3
    latitude = (x - x1) / (x4 - x1) * (f34 - f12) + f12

    return round(latitude, 4), round(longitude, 4)


def dms_to_dd(degrees, minutes, seconds, direction):
    """
    Helper function to convert Degrees Minutes Seconds (DMS) coordinates to Degree Decimal (DD) format.

    :param degrees:   Coordinate's degrees value.
    :param minutes:   Coordinate's minutes value.
    :param seconds:   Coordinate's seconds value.
    :param direction: Direction of the coordinate.
    :return:          The DD version of the DMS coordinate passed as an argument.
    """
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    if direction == 'S' or direction == 'W':
        dd *= -1

    return round(dd, 4)


def dd_to_dms(decimal):
    """
    Helper function to convert Decimal Degrees (DD) coordinates to Degree Minutes Seconds (DMS) format.

    :param decimal: DD value to be converted to DMS format.
    :return:        The DMS equivalent of the DD value passed as an argument.
    """
    degrees = int(decimal)
    sub_decimal = abs(decimal - degrees)
    minutes = int(sub_decimal * 60)
    sub_decimal = abs(sub_decimal - (minutes / 60))
    seconds = round(sub_decimal * 3600, 2)

    return degrees, minutes, seconds


def dd_to_dms_formatter(latitude, longitude):
    """
    Driver function which takes two Degree Decimal (DD) value (one representing a coordinate's latitude, the other
    representing a coordinate's longitude).
    The function converts this tuple to Degrees Minutes Seconds (DMS) format.

    :param latitude:  DD latitude coordinate.
    :param longitude: DD longitude coordinate.
    :return:          The DMS equivalent of the passed DD coordinate.
    """
    latitude_direction  = "N" if latitude >= 0 else "S"
    longitude_direction = "E" if longitude >= 0 else "W"

    latitude_dms  = dd_to_dms(abs(latitude))
    longitude_dms = dd_to_dms(abs(longitude))

    return "{} {}'{}\"{}".format(latitude_dms[0], latitude_dms[1], round(latitude_dms[2], 2), latitude_direction), \
           "{} {}'{}\"{}".format(longitude_dms[0], longitude_dms[1], round(longitude_dms[2], 2), longitude_direction)
