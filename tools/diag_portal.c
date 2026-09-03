/* diag_portal.c — step-by-step CreateSession diagnostic */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sd_bus sd_bus;
typedef struct sd_bus_message sd_bus_message;
typedef struct sd_bus_error {
    const char *name;
    char message[4033]; /* SD_BUS_ERROR_MAX_MESSAGE_LENGTH + 1 */
} sd_bus_error;

static sd_bus_error err_null(void)
{
    sd_bus_error e = { NULL, {0} };
    return e;
}

extern int sd_bus_default_user(sd_bus **bus);
extern int sd_bus_message_new_method_call(sd_bus *bus, sd_bus_message **m,
        const char *destination, const char *path, const char *interface,
        const char *member);
extern int sd_bus_message_open_container(sd_bus_message *m, char type,
        const char *contents);
extern int sd_bus_message_close_container(sd_bus_message *m);
extern int sd_bus_message_append(sd_bus_message *m, const char *types, ...);
extern int sd_bus_call(sd_bus *bus, sd_bus_message *m, uint64_t usec,
        sd_bus_error *ret_error, sd_bus_message **reply);
extern int sd_bus_message_read(sd_bus_message *m, const char *types, ...);
extern void sd_bus_message_unref(sd_bus_message *m);

int main(void)
{
    printf("sizeof(sd_bus_error)=%zu\n", sizeof(sd_bus_error));
    sd_bus *bus = NULL;
    int r = sd_bus_default_user(&bus);
    printf("sd_bus_default_user: r=%d\n", r);
    if (r < 0) return 1;

    sd_bus_message *m = NULL;
    r = sd_bus_message_new_method_call(bus, &m,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.ScreenCast", "CreateSession");
    printf("new_method_call: r=%d m=%p\n", r, (void*)m);

    r = sd_bus_message_open_container(m, 'a', "{sv}");
    printf("open a: r=%d\n", r);
    r = sd_bus_message_open_container(m, 'e', "sv");
    printf("open e: r=%d\n", r);
    r = sd_bus_message_append(m, "s", "token");
    printf("append key: r=%d\n", r);
    r = sd_bus_message_append(m, "s", "diag-test");
    printf("append val: r=%d\n", r);
    r = sd_bus_message_close_container(m);
    printf("close e: r=%d\n", r);
    r = sd_bus_message_close_container(m);
    printf("close a: r=%d\n", r);

    sd_bus_error error = err_null();
    sd_bus_message *reply = NULL;
    r = sd_bus_call(bus, m, 0, &error, &reply);
    printf("sd_bus_call: r=%d err=%s msg=%s\n", r,
           error.name ? error.name : "(null)", error.message);
    sd_bus_message_unref(m);
    if (r < 0)
        return 2;

    char session[512] = {0};
    r = sd_bus_message_read(reply, "o", session);
    printf("read session: r=%d session=%s\n", r, session);
    sd_bus_message_unref(reply);
    printf("DONE\n");
    return 0;
}
