/* Image compare utility. Requires jquery + tabler-icons. */

$(() => {
  $(".image-compare").each((_index, parent) => {
    const $parent = $(parent);
    const before = $parent.data("before-label") || "Before";
    const after = $parent.data("after-label") || "After";
    $parent.append(
      "<div class='image-compare-handle'><i class='ti ti-arrows-horizontal'></i></div>" +
        "<div class='image-compare-before'><div>" +
        before +
        "</div></div>" +
        "<div class='image-compare-after'><div><strong>" +
        after +
        "</strong></div></div>",
    );
  });

  setInterval(() => {
    $(".image-compare").each((_index, parent) => {
      const $parent = $(parent);
      const $handle = $parent.children(".image-compare-handle");

      const currentLeft = $handle.position().left;

      // Linear dynamics + PD controller : - )
      const Kp = 0.03;
      const Kd = 0.2;

      let velocity = $parent.data("velocity") || 0;
      let targetLeft = $parent.data("targetX");
      if (targetLeft !== undefined) {
        const padding = 10;
        const parentWidth = $parent.width();
        if (targetLeft <= padding) targetLeft = 0;
        if (targetLeft >= parentWidth - padding) targetLeft = parentWidth;

        const delta = targetLeft - currentLeft;
        velocity += Kp * delta;
      }
      velocity -= Kd * velocity;

      // Update velocity.
      $parent.data("velocity", velocity);

      const newLeft = currentLeft + velocity;
      $parent.children(".image-compare-handle").css("left", newLeft + "px");
      $parent.children(".image-compare-before").width(newLeft + "px");
      $parent.children("img:not(:first-child)").width(newLeft + "px");
    });
  }, 10);

  $(".image-compare").bind("mousedown touchstart", (evt) => {
    const $parent = $(evt.target.closest(".image-compare"));
    $parent.data("dragging", true);

    if (evt.type == "mousedown")
      $parent.data("targetX", evt.pageX - $parent.offset().left);
    else if (evt.type == "touchstart")
      $parent.data("targetX", evt.touches[0].pageX - $parent.offset().left);
  });

  $(document)
    .bind("mouseup touchend", () => {
      $(".image-compare").each((_index, parent) => {
        $(parent).data("dragging", false);
      });
    })
    .bind("mousemove touchmove", (evt) => {
      $(".image-compare").each((_index, parent) => {
        const $parent = $(parent);
        if (!$parent.data("dragging")) return;

        if (evt.type == "mousemove")
          $parent.data("targetX", evt.pageX - $parent.offset().left);
        else if (evt.type == "touchmove")
          $parent.data("targetX", evt.touches[0].pageX - $parent.offset().left);
      });
    });
});

/* Switcher. */
$(() => {
  $(".switcher").each((switcher_index, switcher) => {
    const $switcher = $(switcher);

    const $inputContainer = $("<div>", { class: "switcher-labels" });

    let $current = null;

    $switcher.children().each((switcher_child_index, child) => {
      const $child = $(child);
      const linkId =
        "switcher-group-" +
        switcher_index.toString() +
        "-" +
        switcher_child_index.toString();
      const $input = $("<input>", {
        type: "radio",
        name: "switcher-group-" + switcher_index.toString(),
        id: linkId,
        checked: switcher_child_index === 0,
        click: function () {
          // Your onclick event logic goes here
          $current.addClass("switcher-hidden");

          $current = $([]);
          $.merge($current, $child);
          $.merge($current, $input);
          $.merge($current, $label);

          $current.removeClass("switcher-hidden");
        },
      });
      const $label = $("<label>", {
        text: $child.data("switcher-label"),
        for: linkId,
      });
      $inputContainer.append($("<div>").append($input).append($label));

      if (switcher_child_index !== 0) {
        $child.addClass("switcher-hidden");
        $input.addClass("switcher-hidden");
        $label.addClass("switcher-hidden");
      } else {
        $current = $([]);
        $.merge($current, $child);
        $.merge($current, $input);
        $.merge($current, $label);
      }
    });

    $switcher.append($inputContainer);
  });
});
