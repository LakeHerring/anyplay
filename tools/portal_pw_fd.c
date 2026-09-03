/*
 * portal_pw_fd.c — xdg-desktop-portal ScreenCast handshake + PipeWire fd handoff.
 *
 * Owns the whole portal/D-Bus side (Python never touches D-Bus):
 *
 *   1. CreateSession(a{sv})              -> request path
 *      wait Request.Response             -> results["session_handle"] (session)
 *   2. SelectSources(session, a{sv})     -> request path
 *      wait Request.Response             -> (KDE: empty; options stored)
 *   3. Start(session, parent_window, a{sv}) -> request path
 *      wait Request.Response             -> results["streams"] a(u@a{sv})
 *      (KDE shows the ScreenChooserDialog here; the user approves)
 *   4. OpenPipeWireRemote(session, a{sv}) -> (h fd)  [synchronous, fd passing]
 *
 * Then either
 *   - execs the command given after `--`, exporting the env contract:
 *         PW_FD             portal PipeWire socket fd (CLOEXEC already cleared)
 *         PW_NODE_ID        first stream's PipeWire node id
 *         PW_PORTAL_SESSION portal session object path
 *         PW_STREAMS        JSON array [{"id":N,"width":W,"height":H}, ...]
 *     The D-Bus session-socket fd is kept open across exec (CLOEXEC cleared)
 *     so the portal session — bound to our unique bus name — stays alive.
 *   - or, with no command, prints the contract as `export ...` lines and
 *     holds the session until Ctrl-C (useful for manual probing).
 *
 * Build:
 *   gcc -O2 -o tools/portal-pw-fd tools/portal_pw_fd.c \
 *       $(pkg-config --cflags --libs gio-2.0)
 *
 * Usage:
 *   ./portal-pw-fd [--types N] [--cursor 0|1] [--token T] [--timeout S] \
 *                  [--] COMMAND [ARGS...]
 *
 *   --types    SelectSources "types" bitmask: 1=monitor 2=window 4=region
 *              (default 1; use 3 to allow monitor+window)
 *   --cursor   SelectSources "cursor_mode": 0=exclude 1=include (default 0)
 *   --timeout  max seconds to wait for the whole handshake (default 300)
 */

#include <gio/gio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>
#include <dirent.h>

/* Clear CLOEXEC on every open fd so the D-Bus session socket (which keeps
 * the portal session alive) and the PipeWire fd survive exec(). */
static void keep_all_fds(void)
{
    int dfd = open("/proc/self/fd", O_RDONLY | O_DIRECTORY);
    if (dfd < 0)
        return;
    DIR *d = fdopendir(dfd);
    struct dirent *e;
    while (d && (e = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)e->d_name[0]))
            continue;
        int f = atoi(e->d_name);
        if (f <= 2)
            continue;
        fcntl(f, F_SETFD, 0);
    }
    if (d)
        closedir(d);
}

#define P_DEST  "org.freedesktop.portal.Desktop"
#define P_PATH  "/org/freedesktop/portal/desktop"
#define P_SC    "org.freedesktop.portal.ScreenCast"
#define P_REQ   "org.freedesktop.portal.Request"

enum { STEP_CREATE = 0, STEP_SELECT, STEP_START, STEP_FD, STEP_DONE };

typedef struct {
    GDBusConnection *bus;
    GMainLoop       *loop;

    /* options */
    uint32_t types;
    uint32_t cursor;
    char     token[128];
    int      timeout_s;
    char    *cmd;
    char   **cmd_argv;

    /* state */
    int     step;
    char    session[512];
    char    request[512];
    int     sub_id;
    char    streams_json[2048];
    int     first_node;
    int     pw_fd;
    int     failed;
    char    fail[256];
} App;

static App app;

/* forward decls */
static gboolean step_select_call(gpointer ud);
static gboolean step_start_call(gpointer ud);
static gboolean step_fd_call(gpointer ud);
static void on_done(void);

static void fail(const char *msg)
{
    if (!app.failed)
        snprintf(app.fail, sizeof app.fail, "%s", msg);
    app.failed = 1;
    if (app.loop)
        g_main_loop_quit(app.loop);
}

static void fail_err(const char *what, GError *e)
{
    if (e) {
        snprintf(app.fail, sizeof app.fail, "%s: %s", what, e->message);
        g_error_free(e);
    } else {
        fail(what);
    }
}

/* ------------------------------------------------------------------ */
/* Request::Response signal                                            */

static void on_response(GDBusConnection *c, const gchar *sender,
                        const gchar *path, const gchar *iface,
                        const gchar *name, GVariant *params, gpointer ud)
{
    (void)c; (void)sender; (void)iface; (void)ud; (void)name;
    if (strcmp(path, app.request) != 0)
        return;

    uint32_t resp = 0;
    GVariant *results = NULL;
    g_variant_get(params, "(u@a{sv})", &resp, &results);
    if (!results) {
        fail("Response: cannot parse (u@a{sv})");
        return;
    }
    if (resp == 1) { /* RESPONSE_CANCELLED */
        fail("request cancelled");
        return;
    }
    if (resp != 0) { /* RESPONSE_FAILURE */
        snprintf(app.fail, sizeof app.fail, "Request::Response error (code %u)", resp);
        app.failed = 1;
        g_main_loop_quit(app.loop);
        return;
    }

    switch (app.step) {
    case STEP_CREATE: {
        const char *sh = NULL;
        if (!g_variant_lookup(results, "session_handle", "s", &sh) || !sh || !*sh) {
            fail("CreateSession Response: missing results[\"session_handle\"]");
            break;
        }
        g_strlcpy(app.session, sh, sizeof app.session);
        fprintf(stderr, "[portal] session: %s\n", app.session);
        app.step = STEP_SELECT;
        break;
    }
    case STEP_SELECT:
        fprintf(stderr, "[portal] SelectSources OK (options stored)\n");
        app.step = STEP_START;
        break;
    case STEP_START: {
        /* This GLib build rejects bare-'a' format strings in g_variant_get
         * (and g_variant_lookup returns TRUE while leaving the output NULL),
         * so unpack via lookup_value + get_child with an exact type and '@'. */
        GVariant *arr = g_variant_lookup_value(
            results, "streams", G_VARIANT_TYPE("a(ua{sv})"));
        if (!arr) {
            fail("Start Response: missing results[\"streams\"]");
            break;
        }
        char *buf = g_malloc(1024);
        buf[0] = 0;
        size_t off = 0;
        int n = 0;
        gsize count = g_variant_n_children(arr);
        for (gsize i = 0; i < count; i++) {
            uint32_t id = 0;
            GVariant *opts = NULL;
            g_variant_get_child(arr, i, "(u@a{sv})", &id, &opts);
            int w = 0, h = 0;
            if (opts)
                g_variant_lookup(opts, "size", "(ii)", &w, &h);
            if (n == 0)
                app.first_node = (int)id;
            off += (size_t)snprintf(buf + off, 1024 - off,
                                    "%s{\"id\":%u,\"width\":%d,\"height\":%d}",
                                    n ? "," : "", id, w, h);
            if (opts)
                g_variant_unref(opts);
            n++;
        }
        g_variant_unref(arr);
        if (n == 0) {
            fail("Start Response: empty streams array");
            g_free(buf);
            break;
        }
        g_strlcpy(app.streams_json, buf, sizeof app.streams_json);
        g_free(buf);
        fprintf(stderr, "[portal] streams: %s\n", app.streams_json);
        app.step = STEP_FD;
        break;
    }
    default:
        break;
    }
    /* Do NOT call the next step (blocking g_dbus_connection_call_sync)
     * directly from this signal callback: in this GLib the nested context
     * iteration does not process the reply while we are still inside the
     * dispatch of the Response signal. Defer to an idle source instead. */
    if (app.step == STEP_SELECT && !app.failed)
        g_idle_add((GSourceFunc)step_select_call, NULL);
    else if (app.step == STEP_START && !app.failed)
        g_idle_add((GSourceFunc)step_start_call, NULL);
    else if (app.step == STEP_FD && !app.failed)
        g_idle_add((GSourceFunc)step_fd_call, NULL);
}

/* ------------------------------------------------------------------ */
/* step calls                                                          */

static void subscribe_request(void)
{
    if (app.sub_id >= 0)
        g_dbus_connection_signal_unsubscribe(app.bus, app.sub_id);
    app.sub_id = g_dbus_connection_signal_subscribe(
        app.bus, P_DEST, P_REQ, "Response", app.request, NULL,
        G_DBUS_SIGNAL_FLAGS_NONE, on_response, NULL, NULL);
}

static const char *call_sync(const char *method, GVariant *args, const char *what)
{
    GError *err = NULL;
    GVariant *reply = g_dbus_connection_call_sync(
        app.bus, P_DEST, P_PATH, P_SC, method, args,
        G_TYPE_INVALID, G_DBUS_CALL_FLAGS_NONE, -1, NULL, &err);
    if (!reply) {
        fail_err(what, err);
        return NULL;
    }
    const char *req = NULL;
    g_variant_get(reply, "(o)", &req);  /* void in this GLib version */
    g_variant_unref(reply);
    fprintf(stderr, "[portal] %s -> request %s (awaiting Response)\n", method, req);
    g_strlcpy(app.request, req, sizeof app.request);
    subscribe_request();
    return req;
}

static void step_create_call(void)
{
    GVariantBuilder ob;
    g_variant_builder_init(&ob, G_VARIANT_TYPE("a{sv}"));
    g_variant_builder_add(&ob, "{sv}",
        "session_handle_token",
        g_variant_new_string(app.token));
    GVariant *opts = g_variant_builder_end(&ob);
    app.step = STEP_CREATE;
    /* '@a{sv}' = embed a GVariant; bare 'a' would consume a GVariantBuilder */
    call_sync("CreateSession", g_variant_new("(@a{sv})", opts), "CreateSession");
}

static gboolean step_select_call(gpointer ud)
{
    (void)ud;
    GVariantBuilder ob;
    g_variant_builder_init(&ob, G_VARIANT_TYPE("a{sv}"));
    g_variant_builder_add(&ob, "{sv}",
        "types",
        g_variant_new_uint32(app.types));
    /* Router validates cursor_mode as a single bitmask:
     * 1=include 2=exclude 4=exclude_region. 0 is rejected.
     * Skip the option entirely if 0 (portal/KDE default). */
    if (app.cursor)
        g_variant_builder_add(&ob, "{sv}",
            "cursor_mode",
            g_variant_new_uint32(app.cursor));
    GVariant *opts = g_variant_builder_end(&ob);
    call_sync("SelectSources",
              g_variant_new("(o@a{sv})", app.session, opts), "SelectSources");
    return G_SOURCE_REMOVE;
}

static gboolean step_start_call(gpointer ud)
{
    (void)ud;
    /* parent_window empty: whole output / portal decides.
     * NULL for the a{sv} slot = empty options dict (special case). */
    call_sync("Start",
              g_variant_new("(osa{sv})", app.session, "", NULL),
              "Start");
    return G_SOURCE_REMOVE;
}

static gboolean step_fd_call(gpointer ud)
{
    (void)ud;
    fprintf(stderr, "[portal] step_fd_call: OpenPipeWireRemote %s\n", app.session);
    /* Use the raw-message API: this GLib build returns plain GVariants
     * from call_sync (not GVariantGDBus), so G_DBUS_MESSAGE(reply) and
     * the attached fd list are unreachable through the GVariant. */
    GError *err = NULL;
    GDBusMessage *req = g_dbus_message_new_method_call(
        P_DEST, P_PATH, P_SC, "OpenPipeWireRemote");
    g_dbus_message_set_body(req, g_variant_new("(oa{sv})", app.session, NULL));
    GDBusMessage *reply = g_dbus_connection_send_message_with_reply_sync(
        app.bus, req, 0, -1, NULL, NULL, &err);
    g_object_unref(req);
    if (!reply) {
        fail_err("OpenPipeWireRemote", err);
        return G_SOURCE_REMOVE;
    }
    GUnixFDList *fl = g_dbus_message_get_unix_fd_list(reply);
    if (!fl) {
        fail("OpenPipeWireRemote: no fd list in reply");
        g_object_unref(reply);
        return G_SOURCE_REMOVE;
    }
    int fd_id = -1;
    g_variant_get(g_dbus_message_get_body(reply), "(h)", &fd_id);
    int fd = g_unix_fd_list_get(fl, fd_id, NULL);
    if (fd < 0) {
        fail("OpenPipeWireRemote: bad fd id");
        g_object_unref(reply);
        return G_SOURCE_REMOVE;
    }
    /* IMPORTANT: do NOT unref `reply`/`fl`. GLib owns the fds in the
     * fd-list; freeing the message would close the PipeWire socket
     * before (or after) exec. Deliberate leak — one fd for process
     * lifetime. */
    app.pw_fd = fd;
    fprintf(stderr, "[portal] PipeWire fd: %d\n", fd);
    app.step = STEP_DONE;
    on_done();
    return G_SOURCE_REMOVE;
}

/* ------------------------------------------------------------------ */
/* done: exec or hold                                                  */

static gboolean on_timeout(gpointer ud)
{
    (void)ud;
    fail("timeout waiting for portal handshake (user approval?)");
    return G_SOURCE_REMOVE;
}

static void on_done(void)
{
    if (!app.cmd) {
        printf("export PW_FD=%d\n", app.pw_fd);
        printf("export PW_NODE_ID=%d\n", app.first_node);
        printf("export PW_PORTAL_SESSION=%s\n", app.session);
        printf("export PW_STREAMS=%s\n", app.streams_json);
        fflush(stdout);
        fprintf(stderr, "[portal] session held (Ctrl-C to end); "
                        "consumer should adopt fd %d\n", app.pw_fd);
        return; /* keep the main loop running: the session dies with us */
    }

    /* --- env contract --- */
    char buf[64];
    snprintf(buf, sizeof buf, "%d", app.pw_fd);
    setenv("PW_FD", buf, 1);
    snprintf(buf, sizeof buf, "%d", app.first_node);
    setenv("PW_NODE_ID", buf, 1);
    setenv("PW_PORTAL_SESSION", app.session, 1);
    setenv("PW_STREAMS", app.streams_json, 1);

    /* Keep the D-Bus session socket alive across exec: the portal session
     * is bound to our unique bus name; if the socket fd dies, the router
     * closes the session and the streams vanish. */
    keep_all_fds();

    fprintf(stderr, "[portal] exec: %s (PW_FD=%d PW_NODE_ID=%d)\n",
            app.cmd, app.pw_fd, app.first_node);
    fflush(stderr);
    execvp(app.cmd, app.cmd_argv);
    fprintf(stderr, "[portal] execvp(%s) failed: %s\n", app.cmd, strerror(errno));
    exit(127);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    memset(&app, 0, sizeof app);
    app.types = 1;          /* monitor */
    app.cursor = 1;          /* 1=include (router rejects 0; 2=exclude) */
    app.timeout_s = 300;
    snprintf(app.token, sizeof app.token, "anyplay_%ld", (long)time(NULL));
    app.pw_fd = -1;
    app.first_node = -1;
    app.sub_id = -1;

    int i = 1;
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--")) { i++; break; }
        else if (!strcmp(argv[i], "--types") && i + 1 < argc)
            app.types = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--cursor") && i + 1 < argc)
            app.cursor = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--token") && i + 1 < argc)
            g_strlcpy(app.token, argv[++i], sizeof app.token);
        else if (!strcmp(argv[i], "--timeout") && i + 1 < argc)
            app.timeout_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            fprintf(stderr,
                "usage: %s [--types N] [--cursor 0|1|2|4] [--token T] [--timeout S] [--] CMD...\n"
                "  types bitmask: 1=monitor 2=window 4=region (default 1)\n"
                "  cursor: 0=omit 1=include 2=exclude 4=exclude_region (default 1)\n"
                "  without CMD: prints export lines and holds the session\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (i < argc) {
        app.cmd = argv[i];
        app.cmd_argv = &argv[i];
    }

    app.bus = g_bus_get_sync(G_BUS_TYPE_SESSION, NULL, NULL);
    if (!app.bus) {
        fprintf(stderr, "[portal] cannot get session bus (DBUS_SESSION_BUS_ADDRESS?)\n");
        return 1;
    }
    app.loop = g_main_loop_new(NULL, FALSE);
    g_timeout_add(app.timeout_s * 1000, on_timeout, NULL);

    step_create_call();
    g_main_loop_run(app.loop);

    if (app.failed) {
        fprintf(stderr, "[portal] FAILED: %s\n", app.fail);
        return 1;
    }
    return 0;
}
