from Geometry import HtmlElement, HtmlHead


if __name__ == "__main__":
    #  page = HtmlElement("!DOCTYPE html")
    html = HtmlElement("html", "html") # , parent=page)
    head = HtmlHead(parent=html)
    style = head.append_style()
    head.append_script()
    head.append_meta()
    head.append_link()
    style.add_child(HtmlElement("div", "div"))
    # head.add_child(HtmlElement("h1", "h1", inner_html="PageHeader", class_list=("header-1", "main")))
    # title = HtmlElement("title", "title", parent=head)
    # h_style = title.style
    # h_style.width = 1200
    # h_style.height = 50
    # h_style.padding = (CssUnit(1, CssUnits.CM), CssUnit(2, CssUnits.INCH), CssUnit(3, CssUnits.PERCENT), CssUnit(4, CssUnits.AUTO))
    body = HtmlElement("body", "body", parent=html)
    print(html)
