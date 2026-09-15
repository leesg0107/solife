---
layout: page
title: Study
subtitle: Notes on reinforcement learning, multi-robot systems, and robotics
permalink: /study/
---

<div class="posts-list">
{% assign study_posts = site.posts | where: "category", "study" %}
{% if study_posts.size > 0 %}
  {% for post in study_posts %}
  <article>
    <h2 class="post-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>

    {% if post.subtitle %}
    <p class="post-subtitle">{{ post.subtitle }}</p>
    {% endif %}

    <p class="post-meta">{{ post.date | date: "%B %d, %Y" }}</p>

    {% if post.tags %}
    <div class="post-tags">
      {% for tag in post.tags limit: 5 %}
      <span class="post-tag">{{ tag }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </article>
  {% endfor %}
{% else %}
  <p><em>No study posts yet.</em></p>
{% endif %}
</div>
